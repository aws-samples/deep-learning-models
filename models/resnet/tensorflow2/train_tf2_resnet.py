# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''

        
mpirun -np 8 --H localhost:8 --allow-run-as-root \
        -mca btl_tcp_if_exclude docker0,lo -x NCCL_SOCKET_IFNAME=^docker0,lo \
        python -W ignore deep-learning-models/models/resnet/tensorflow2/train_tf2_resnet.py \
        --synthetic \
        --batch_size 256 --num_batches 100 --clear_log 2
        
mpirun -np 8 --H localhost:8 --allow-run-as-root \
        -mca btl_tcp_if_exclude docker0,lo -x NCCL_SOCKET_IFNAME=^docker0,lo \
        python -W ignore deep-learning-models/models/resnet/tensorflow2/train_tf2_resnet.py \
        --data_dir /workspace/shared_workspace/data \
        --batch_size 256 --num_batches 100 --clear_log 2
'''




import os
import numpy as np
import getpass
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.util import nest
import argparse
from time import time, sleep

@tf.function
def parse(record):
    features = {'image/encoded': tf.io.FixedLenFeature((), tf.string),
                'image/class/label': tf.io.FixedLenFeature((), tf.int64)}
    parsed = tf.io.parse_single_example(record, features)
    image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float16)
    label = tf.cast(parsed['image/class/label'] - 1, tf.int32)
    return image, label

def data_gen():
    input_shape = [224, 224, 3]
    while True:
        image = tf.random.uniform(input_shape)
        label = tf.random.uniform(minval=0, maxval=999, shape=[1], dtype=tf.int32)
        yield image, label

def create_data(data_dir = None, synthetic=False, batch_size=128):
    if synthetic:
        ds = tf.data.Dataset.from_generator(data_gen, output_types=(tf.float32, tf.int32))
    else:
        filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        ds = tf.data.TFRecordDataset(filenames).shard(hvd.size(), hvd.rank())
        ds = ds.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(128)
    return ds

@tf.function
def train_step(model, opt, loss_func, images, labels, first_batch, fp32=False):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss_func(labels, probs)
        if not fp32:
            scaled_loss = opt.get_scaled_loss(loss_value)
    tape = hvd.DistributedGradientTape(tape, compression=hvd.Compression.fp16)
    if fp32:
        grads = tape.gradient(loss_value, model.trainable_variables)
    else:
        scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads = opt.get_unscaled_gradients(scaled_grads)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return loss_value

def add_cli_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--data_dir', default='',
                         help="""Path to dataset in TFRecord format
                             (aka Example protobufs). Files should be
                             named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('-b', '--batch_size', default=128, type=int,
                         help="""Size of each minibatch per GPU""")
    cmdline.add_argument('--num_batches', default=100, type=int,
                         help="""Number of batches to run.
                             Ignored during eval or if num epochs given""")
    cmdline.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                         help="""Start learning rate""")
    cmdline.add_argument('--momentum', default=0.01, type=float,
                         help="""Start learning rate""")
    cmdline.add_argument('-fp32', '--fp32', 
                         help="""disable mixed precision training""",
                         action='store_true')
    cmdline.add_argument('-xla_off', '--xla_off', 
                         help="""disable xla""",
                         action='store_true')
    cmdline.add_argument('-s', '--synthetic', 
                         help="""Use synthetic data for training""",
                         action='store_true')
    return cmdline

def main():
    # setup horovod
    start = time()
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()
    if not FLAGS.xla_off:
        tf.config.optimizer.set_jit(True)
    if not FLAGS.fp32:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    ds = create_data(FLAGS.data_dir, FLAGS.synthetic, FLAGS.batch_size)
    model = tf.keras.applications.ResNet50(weights=None, classes=1000)
    opt = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate * hvd.size(), momentum=0.1)
    if not FLAGS.fp32:
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, loss_scale="dynamic")
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    loop_time = time()
    if hvd.local_rank() == 0:
        print("Step \t Throughput \t Loss")
    for batch, (images, labels) in enumerate(ds):
        loss = train_step(model, opt, loss_func, images, labels, batch==0, fp32=FLAGS.fp32)
        if hvd.local_rank() == 0:
            duration = time() - loop_time
            loop_time = time()
            throughput = (hvd.size()*FLAGS.batch_size)/duration
            print("{} \t images/sec: {} \t {}".format(batch, throughput, loss))
        if batch==FLAGS.num_batches:
            break
    if hvd.rank() == 0:
        print("\nFinished in {}".format(time()-start))

if __name__=='__main__':
    main()
