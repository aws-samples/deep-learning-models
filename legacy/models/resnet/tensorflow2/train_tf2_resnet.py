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
mpirun -np 8 --H localhost:8 \
-bind-to none -map-by slot -mca pml ob1 -mca -x TF_CUDNN_USE_AUTOTUNE=0 \
-x TF_ENABLE_NHWC=1 -x FI_OFI_RXR_INLINE_MR_ENABLE=1 -x NCCL_TREE_THRESHOLD=4294967296 \
-x PATH -x NCCL_SOCKET_IFNAME=^docker0,lo -x NCCL_MIN_NRINGS=13 -x NCCL_DEBUG=INFO \
-x HOROVOD_CYCLE_TIME=0.5 -x HOROVOD_FUSION_THRESHOLD=67108864 python new_resnet.py --synthetic
source activate tensorflow2_p36 && \
mpirun -np 8 --H localhost:8 -mca plm_rsh_no_tree_spawn 1 \
        -bind-to socket -map-by slot \
        -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
        -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
        -x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_exclude lo,docker0 \
        -x TF_CPP_MIN_LOG_LEVEL=0 \
        python -W ignore ~/new_resnet.py \
        --synthetic --batch_size 128 --num_batches 100 --clear_log 2 > train.log
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
    image = tf.image.random_brightness(image, .1)
    image = tf.image.random_jpeg_quality(image, 70, 100)
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed['image/class/label'] - 1, tf.int32)
    return image, label

def data_gen():
    input_shape = [224, 224, 3]
    while True:
        image = tf.random.uniform(input_shape)
        label = tf.random.uniform(minval=0, maxval=999, shape=[1], dtype=tf.int32)
        yield image, label

def create_data(data_dir = None, synthetic=False, batch_size=256):
    if synthetic:
        ds = tf.data.Dataset.from_generator(data_gen, output_types=(tf.float32, tf.int32))
    else:
        filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        ds = tf.data.Dataset.from_tensor_slices(filenames).shard(hvd.size(), hvd.rank())
        ds = ds.shuffle(1000, seed=7 * (1 + hvd.rank()))
        ds = ds.interleave(
            tf.data.TFRecordDataset, cycle_length=1, block_length=1)
        ds = ds.map(parse, num_parallel_calls=10)
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(10000, seed=5 * (1 + hvd.rank())))
    ds = ds.batch(batch_size)
    return ds

@tf.function
def train_step(model, opt, loss_func, images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss_func(labels, probs)
    tape = hvd.DistributedGradientTape(tape, compression=hvd.Compression.fp16)
    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return loss_value

def add_bool_argument(cmdline, shortname, longname=None, default=False, help=None):
    if longname is None:
        shortname, longname = None, shortname
    elif default == True:
        raise ValueError("""Boolean arguments that are True by default should not have short names.""")
    name = longname[2:]
    feature_parser = cmdline.add_mutually_exclusive_group(required=False)
    if shortname is not None:
        feature_parser.add_argument(shortname, '--' + name, dest=name, action='store_true', help=help, default=default)
    else:
        feature_parser.add_argument('--' + name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no' + name, dest=name, action='store_false')
    return cmdline

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
    add_bool_argument(cmdline, '--synthetic', help="""Whether to use synthetic data for training""")
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
    # get command line args
    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()
    ds = create_data(FLAGS.data_dir, FLAGS.synthetic, FLAGS.batch_size)
    model = tf.keras.applications.ResNet50(weights=None, classes=1000)
    opt = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate * hvd.size(), momentum=0.1)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    loop_time = time()
    if hvd.local_rank() == 0:
        print("Step \t Throughput \t Loss")
    for batch, (images, labels) in enumerate(ds):
        loss = train_step(model, opt, loss_func, images, labels, batch==0)
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