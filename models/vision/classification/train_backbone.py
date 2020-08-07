import os
import functools
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
from models import resnet, darknet, hrnet
from schedulers import WarmupScheduler
from datasets import create_dataset, parse
from trainer import train_step, validation_step
from optimizers import MomentumOptimizer

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
    cmdline.add_argument('--num_classes', default=1000, type=int,
                         help="""Number of classes.""")
    cmdline.add_argument('--train_dataset_size', default=1281167, type=int,
                         help="""Number of images in training data.""")
    cmdline.add_argument('--model_dir', default='checkpoints',
                         help="""Path to save model with best accuracy""")
    cmdline.add_argument('-b', '--batch_size', default=128, type=int,
                         help="""Size of each minibatch per GPU""")
    cmdline.add_argument('--num_epochs', default=120, type=int,
                         help="""Number of epochs to train for.""")
    cmdline.add_argument('--schedule', default='cosine', type=str,
                         help="""learning rate schedule""")
    cmdline.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                         help="""Start learning rate.""")
    cmdline.add_argument('--momentum', default=0.9, type=float,
                         help="""Start optimizer momentum.""")
    cmdline.add_argument('--label_smoothing', default=0.1, type=float,
                         help="""Label smoothing value.""")
    cmdline.add_argument('--mixup_alpha', default=0.2, type=float,
                        help="""Mixup beta distribution shape parameter. 0.0 disables mixup.""")
    cmdline.add_argument('--l2_weight_decay', default=1e-4, type=float,
                         help="""L2 weight decay multiplier.""")
    cmdline.add_argument('-fp32', '--fp32', 
                         help="""disable mixed precision training""",
                         action='store_true')
    cmdline.add_argument('-xla_off', '--xla_off', 
                         help="""disable xla""",
                         action='store_true')
    cmdline.add_argument('--model',
                         help="""Which model to train. Options are:
                         resnet50v1_b, resnet50v1_c, resnet50v1_d, resnet101v1_b, resnet101v1_c,resnet101v1_d, darknet53, hrnet_w18c, hrnet_w32c""")
    cmdline.add_argument('--fine_tune',
                         help="""Whether to fine tune a pretrained model or 
                         train the full model from scratch. Must specify weights
                         path if flag is set.""",
                         action='store_true', default=False)
    cmdline.add_argument('--resume_from', 
                         help='Path to SavedModel format model directory from which to resume training')
    return cmdline


def main():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.threading.intra_op_parallelism_threads = 1 # Avoid pool of Eigen threads
    tf.config.threading.inter_op_parallelism_threads = max(2, 40//hvd.size()-2)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()

    if FLAGS.fine_tune:
        raise NotImplementedError('fine tuning functionality not available')

    if not FLAGS.xla_off:
        tf.config.optimizer.set_jit(True)
    if not FLAGS.fp32:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    preprocessing_type = 'resnet'
    if FLAGS.model == 'resnet50v1_b':
        model = resnet.ResNet50V1_b(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet50v1_c':
        model = resnet.ResNet50V1_c(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet50v1_d':
        model = resnet.ResNet50V1_d(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet101v1_b':
        model = resnet.ResNet101V1_b(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet101v1_c':
        model = resnet.ResNet101V1_c(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'resnet101v1_d':
        model = resnet.ResNet101V1_d(weights=None, weight_decay=FLAGS.l2_weight_decay, classes=FLAGS.num_classes)
    elif FLAGS.model == 'darknet53':
        model = darknet.Darknet(weight_decay=FLAGS.l2_weight_decay)
    elif FLAGS.model in ['hrnet_w18c', 'hrnet_w32c']:
        preprocessing_type = 'imagenet'
        model = hrnet.build_hrnet(FLAGS.model)
        model._set_inputs(tf.keras.Input(shape=(None, None, 3)))
    else:
        raise NotImplementedError('Model {} not implemented'.format(FLAGS.model))

    model.summary()

    # scale learning rate linearly, base learning rate for batch size of 256 is specified through args
    BASE_LR = FLAGS.learning_rate
    learning_rate = (BASE_LR * hvd.size() * FLAGS.batch_size)/256 
    steps_per_epoch = int((FLAGS.train_dataset_size / (FLAGS.batch_size * hvd.size())))

    # 5 epochs are for warmup
    if FLAGS.schedule == 'piecewise_short':
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[steps_per_epoch * 25, steps_per_epoch * 55, steps_per_epoch * 75, step_per_epoch * 100], 
                    values=[learning_rate, learning_rate * 0.1, learning_rate * 0.01, learning_rate * 0.001, learning_rate * 0.0001])
    elif FLAGS.schedule == 'piecewise_long':
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=[steps_per_epoch * 55, steps_per_epoch * 115, steps_per_epoch * 175], 
                    values=[learning_rate, learning_rate * 0.1, learning_rate * 0.01, learning_rate * 0.001])
    elif FLAGS.schedule == 'cosine':
        scheduler = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=learning_rate,
                    first_decay_steps=FLAGS.num_epochs*steps_per_epoch, t_mul=1, m_mul=1)
    else:
        print('No schedule specified')


    scheduler = WarmupScheduler(optimizer=scheduler, initial_learning_rate=learning_rate / hvd.size(), warmup_steps=steps_per_epoch * 5)

    #TODO support optimizers choice via config
    # opt = tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=FLAGS.momentum, nesterov=True) # needs momentum correction term
    opt = MomentumOptimizer(learning_rate=scheduler, momentum=FLAGS.momentum, nesterov=True) 

    if not FLAGS.fp32:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale=128.)


    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=FLAGS.label_smoothing, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) 

    if hvd.rank() == 0:
        if FLAGS.resume_from:
            model = tf.keras.models.load_model(FLAGS.resume_from)
            print('loaded model from', FLAGS.resume_from)
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
        logger.info('Training options: %s', FLAGS)
    
    # barrier
    hvd.allreduce(tf.constant(0))
 
    start_time = time()
    curr_step = tf.Variable(initial_value=0, dtype=tf.int32)
    best_validation_accuracy = 0.7 # only save 0.7 or higher checkpoints
    
    data = create_dataset(FLAGS.train_data_dir, FLAGS.batch_size, preprocessing=preprocessing_type, validation=False)
    validation_data = create_dataset(FLAGS.validation_data_dir, FLAGS.batch_size, preprocessing=preprocessing_type, validation=True)

    for epoch in range(FLAGS.num_epochs):
        if hvd.rank() == 0:
            print('Starting training Epoch %d/%d' % (epoch, FLAGS.num_epochs))
        training_score = 0
        for batch, (images, labels) in enumerate(tqdm(data)):
            # momentum correction (V2 SGD absorbs LR into the update term)
            # prev_lr = opt._optimizer.learning_rate(curr_step-1)
            # curr_lr = opt._optimizer.learning_rate(curr_step)
            # momentum_correction_factor = curr_lr / prev_lr
            # opt._optimizer.momentum = opt._optimizer.momentum * momentum_correction_factor
            loss, score = train_step(model, opt, loss_func, images, labels, batch==0 and epoch==0,
                            batch_size=FLAGS.batch_size, mixup_alpha=FLAGS.mixup_alpha, fp32=FLAGS.fp32)
            # # restore momentum
            # opt._optimizer.momentum = FLAGS.momentum
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
            info_str = 'Epoch: %d, Train Accuracy: %f, Train Loss: %f, Validation Accuracy: %f, Validation Loss: %f LR:%f' % (
                    epoch, average_training_accuracy, average_training_loss, average_validation_accuracy, average_validation_loss, scheduler(curr_step))
            print(info_str)
            logger.info(info_str)
            if average_validation_accuracy > best_validation_accuracy:
                logger.info("Found new best accuracy, saving checkpoint ...")
                best_validation_accuracy = average_validation_accuracy
                model.save('{}/{}'.format(FLAGS.model_dir, FLAGS.model))
    if hvd.rank() == 0:
        logger.info('Total Training Time: %f' % (time() - start_time))

if __name__ == '__main__':
    main()

