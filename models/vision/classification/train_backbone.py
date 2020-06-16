import os
import tensorflow as tf
import tensorflow_addons as tfa
import horovod.tensorflow as hvd
import numpy as np
import argparse
import datetime
import random
import logging
from tqdm import tqdm
from time import time
import sys
import resnet_preprocessing
import resnet


class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Wraps another learning rate scheduler to add a linear or exponential warmup
    """
    
    def __init__(self, optimizer, initial_learning_rate, warmup_steps, warmup_type='linear',
                 dtype=tf.float32):
        super(WarmupScheduler, self).__init__()
        self.optimizer = optimizer
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype)
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.warmup_type = warmup_type
        self.dtype = dtype
        self.optimizer_learning_rate = optimizer(0)
        
    def compute_linear_warmup(self, step):
        return ((self.optimizer_learning_rate*step) + (self.initial_learning_rate*(self.warmup_steps-step)))/self.warmup_steps
    
    @tf.function(experimental_relax_shapes=True)
    def __call__(self, step):
        global_step_recomp = tf.cast(step, self.dtype)
        if global_step_recomp>=self.warmup_steps:
            return self.optimizer(global_step_recomp - self.warmup_steps)
        return self.compute_linear_warmup(global_step_recomp)
    
    def get_config(self):
        optimizer_config = self.optimizer.get_config()
        optimizer_config['initial_learning_rate'] = self.initial_learning_rate
        optimizer_config['warmup_steps'] = self.warmup_steps
        optimizer_config['warmup_type'] = self.warmup_type


@tf.function
def parse(record, is_training):
    features = {'image/encoded': tf.io.FixedLenFeature((), tf.string),
                'image/class/label': tf.io.FixedLenFeature((), tf.int64),
                'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                }
    parsed = tf.io.parse_single_example(record, features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    # bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox = tf.stack([parsed['image/object/bbox/%s' % x].values for x in ['ymin', 'xmin', 'ymax', 'xmax']])
    bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
    image = resnet_preprocessing.preprocess_image(image_bytes, bbox, 224, 224, 3, is_training=is_training)
    label = tf.cast(parsed['image/class/label'] - 1, tf.int32)
    one_hot_label = tf.one_hot(label, depth=1000, dtype=tf.int32)
    return image, one_hot_label

def parse_train(record):
    return parse(record, is_training=True)

def parse_validation(record):
    return parse(record, is_training=False)

def add_cli_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--train_data_dir', default='',
                         help="""Path to dataset in TFRecord format
                             (aka Example protobufs). Files should be
                             named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('--validation_data_dir', default='',
                         help="""Path to dataset in TFRecord format
                             (aka Example protobufs). Files should be
                             named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('-b', '--batch_size', default=256, type=int,
                         help="""Size of each minibatch per GPU""")
    cmdline.add_argument('--num_epochs', default=100, type=int,
                         help="""Number of epochs to train for.""")
    cmdline.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                         help="""Start learning rate""")
    cmdline.add_argument('--momentum', default=0.9, type=float,
                         help="""Start optimizer momentum""")
    cmdline.add_argument('-fp32', '--fp32', 
                         help="""disable mixed precision training""",
                         action='store_true')
    cmdline.add_argument('-xla_off', '--xla_off', 
                         help="""disable xla""",
                         action='store_true')
    cmdline.add_argument('--model',
                         help="""Which model to train. Options are:
                         resnet50 and resnext50""")
    cmdline.add_argument('--fine_tune',
                         help="""Whether to fine tune on pretrained model or 
                         train the full model from scratch. Must specify weights
                         path if flag is set.""",
                         action='store_true')
    cmdline.add_argument('--weights_path', 
                         help='Path to weights for pretrained model')
    return cmdline

def create_dataset(data_dir, batch_size, validation):
    filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
    data = tf.data.TFRecordDataset(filenames).shuffle(buffer_size=10000).shard(hvd.size(), hvd.rank())
    if not validation:
        data = data.map(parse_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        data = data.map(parse_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=not validation).prefetch(tf.data.experimental.AUTOTUNE)
    return data

@tf.function
def train_step(model, opt, loss_func, images, labels, first_batch, fp32=False):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss_func(labels, probs)
        loss_value += tf.add_n(model.losses)
        if not fp32:
            scaled_loss_value = opt.get_scaled_loss(loss_value)

    tape = hvd.DistributedGradientTape(tape, compression=hvd.Compression.fp16)
    if not fp32:
        grads = tape.gradient(scaled_loss_value, model.trainable_variables)
        grads = opt.get_unscaled_gradients(grads)
    else:
        grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    top_1_pred = tf.squeeze(tf.math.top_k(probs, k=1)[1])
    sparse_labels = tf.cast(tf.math.argmax(labels, axis=1), tf.int32)
    top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, sparse_labels), tf.int32))
    # top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, labels), tf.int32))
    return loss_value, top_1_accuracy

@tf.function
def validation_step(images, labels, model, loss_func):
    pred = model(images, training=False)
    loss = loss_func(labels, pred)
    top_1_pred = tf.squeeze(tf.math.top_k(pred, k=1)[1])
    sparse_labels = tf.cast(tf.math.argmax(labels, axis=1), tf.int32)
    top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, sparse_labels), tf.int32))
    # top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, labels), tf.int32))
    return loss, top_1_accuracy



def main():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.threading.intra_op_parallelism_threads = 1 # Avoid pool of Eigen threads
    tf.config.threading.inter_op_parallelism_threads = max(2, 96//hvd.size()-2)

    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()
    if not FLAGS.xla_off:
        tf.config.optimizer.set_jit(True)
    if not FLAGS.fp32:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    # init global imagenet mean (for tf function graph)
    # temp = tf.zeros([128, 224, 224, 3], dtype=tf.float32)
    # _ = tf.keras.applications.resnet.preprocess_input(temp)

    data = create_dataset(FLAGS.train_data_dir, FLAGS.batch_size, validation=False)
    validation_data = create_dataset(FLAGS.validation_data_dir, FLAGS.batch_size, validation=True)

    if FLAGS.model == 'resnet50':
        if not FLAGS.fine_tune:
            model = resnet.ResNet50(weights=None, weight_decay=0.00005, pooling='avg', classes=1000)
        else:
            model = resnet.ResNet50(weights='imagenet', classes=1000)
    model.summary()
    learning_rate = (FLAGS.learning_rate * hvd.size() * FLAGS.batch_size)/256 
    steps_per_epoch = int((1281167 / (FLAGS.batch_size * hvd.size())))

    scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[steps_per_epoch * 25, steps_per_epoch * 55, steps_per_epoch * 75], # 5 epochs for warmup
                    values=[learning_rate, learning_rate * 0.1, learning_rate * 0.01, learning_rate * 0.001])

    scheduler = WarmupScheduler(optimizer=scheduler, initial_learning_rate=learning_rate / hvd.size(), warmup_steps=steps_per_epoch * 5)
    # opt = tfa.optimizers.SGDW(weight_decay=1e-5, learning_rate=scheduler, momentum=FLAGS.momentum, nesterov=True)
    opt = tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=FLAGS.momentum, nesterov=True) # FIXME: not correct - needs momentum correction term
    # opt = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr_func, momentum=0.9, use_nesterov=True)

    if not FLAGS.fp32:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale=128.)

    loss_func = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) 
    # loss_func = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) 
    # loss_func = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    if hvd.rank() == 0:
        model_dir = os.path.join(FLAGS.model + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"))
        path_logs = os.path.join(os.getcwd(), model_dir, 'log.csv')
        os.mkdir(model_dir)
        logging.basicConfig(filename=path_logs,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        logging.info("Training Logs")
        logger = logging.getLogger('logger')
        logger.info('Batch Size: %f, Learning Rate: %f, Momentum: %f' % \
                    (FLAGS.batch_size, FLAGS.learning_rate, FLAGS.momentum))

    hvd.allreduce(tf.constant(0))
 
    start_time = time()
    curr_step = tf.Variable(initial_value=0, dtype=tf.int32)
    for epoch in range(FLAGS.num_epochs):
        if hvd.rank() == 0:
            print('Starting training Epoch %d/%d' % (epoch, FLAGS.num_epochs))
        training_score = 0
        for batch, (images, labels) in enumerate(tqdm(data)):
            # momentum correction (V2 SGD absorbs LR into the update term)
            prev_lr = opt._optimizer.learning_rate(curr_step-1)
            curr_lr = opt._optimizer.learning_rate(curr_step)
            momentum_correction_factor = curr_lr / prev_lr
            opt._optimizer.momentum = opt._optimizer.momentum * momentum_correction_factor
            loss, score = train_step(model, opt, loss_func, images, labels, batch==0 and epoch==0, fp32=FLAGS.fp32)
            # restore momentum
            opt._optimizer.momentum = FLAGS.momentum
            training_score += score.numpy()
            curr_step.assign_add(1)
        training_accuracy = training_score / (FLAGS.batch_size * (batch + 1))
        average_training_accuracy = hvd.allreduce(tf.constant(training_accuracy))
        average_training_loss = hvd.allreduce(tf.constant(loss))

        if hvd.rank() == 0:
            print('Starting validation Epoch %d/%d' % (epoch, FLAGS.num_epochs))
        validation_score = 0
        counter = 0
        for images, labels in tqdm(validation_data):
            loss, score = validation_step(images, labels, model, loss_func)
            validation_score += score.numpy()
            counter += 1
        validation_accuracy = validation_score / (FLAGS.batch_size * counter)
        average_validation_accuracy = hvd.allreduce(tf.constant(validation_accuracy))
        average_validation_loss = hvd.allreduce(tf.constant(loss))

        if hvd.rank() == 0:
            #path_checkpoint = path_logs = os.path.join(os.getcwd(), model_dir)
            info_str = 'Epoch: %d, Train Accuracy: %f, Train Loss: %f, Validation Accuracy: %f, Validation Loss: %f LR:%f' % \
                    (epoch, average_training_accuracy, average_training_loss, average_validation_accuracy, average_validation_loss, scheduler(curr_step))
            print(info_str)
            logger.info(info_str)
            #checkpoint.save(path_checkpoint)
    if hvd.rank() == 0:
        logger.info('Total Training Time: %f' % (time() - start_time))
        model.save('saved_model/resnet50')

if __name__ == '__main__':
    main()

