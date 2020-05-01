#!/usr/bin/env python
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script takes an existing tfrecord dataset and generates a new one
# with the images resized.
# E.g.,
# python tensorflow_image_resizer.py \
#     -i /path/to/imagenet-full-tfrecord/ -o /path/to/imagenet-new-tfrecord/ --subset_name train
# python tensorflow_image_resizer.py \
#     -i /path/to/imagenet-full-tfrecord/ -o /path/to/imagenet-new-tfrecord/ --subset_name validation

from __future__ import print_function
from builtins import range
from multiprocessing import cpu_count
import os
import tensorflow as tf
import time

global FLAGS

class Dataset(object):
    def __init__(self, name, data_dir=None):
        self.name = name
        if data_dir is None:
            data_dir = FLAGS.data_dir
        self.data_dir = data_dir
    def data_files(self, subset):
        tf_record_pattern = os.path.join(self.data_dir, '%s-*' % subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            raise RuntimeError('No files found for %s dataset at %s' %
                               (subset, self.data_dir))
        return data_files
    def reader(self):
        return tf.TFRecordReader()
    def num_classes(self):
        raise NotImplementedError
    def num_examples_per_epoch(self, subset):
        raise NotImplementedError
    def __str__(self):
        return self.name

class ImagenetData(Dataset):
    def __init__(self, data_dir=None):
        super(ImagenetData, self).__init__('ImageNet', data_dir)
    def num_classes(self):
        return 1000
    def num_examples_per_epoch(self, subset):
        if   subset == 'train':      return 1281167
        elif subset == 'validation': return 50000
        else: raise ValueError('Invalid data subset "%s"' % subset)

class FlowersData(Dataset):
    def __init__(self, data_dir=None):
        super(FlowersData, self).__init__('Flowers', data_dir)
    def num_classes(self):
        return 5
    def num_examples_per_epoch(self, subset):
        if   subset == 'train':      return 3170
        elif subset == 'validation': return 500
        else: raise ValueError('Invalid data subset "%s"' % subset)

def resize_example(example):
    # Dense features in Example proto.
    feature_map = {
        'image/encoded':      tf.FixedLenFeature([],  dtype=tf.string, default_value=''),
        'image/height':       tf.FixedLenFeature([1], dtype=tf.int64,  default_value=-1),
        'image/width':        tf.FixedLenFeature([1], dtype=tf.int64,  default_value=-1),
        'image/channels':     tf.FixedLenFeature([1], dtype=tf.int64,  default_value=-1),
        'image/colorspace':   tf.FixedLenFeature([],  dtype=tf.string, default_value=''),
        'image/class/label':  tf.FixedLenFeature([1], dtype=tf.int64,  default_value=-1),
        'image/class/text':   tf.FixedLenFeature([],  dtype=tf.string, default_value=''),
        'image/class/synset': tf.FixedLenFeature([],  dtype=tf.string, default_value=''),
        'image/format':       tf.FixedLenFeature([],  dtype=tf.string, default_value=''),
        'image/filename':     tf.FixedLenFeature([],  dtype=tf.string, default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        #{k: sparse_float32 for k in ['image/object/bbox/xmin',
        {k: tf.VarLenFeature(dtype=tf.float32) for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})
    example = tf.parse_single_example(example, feature_map)
    encoded_image = example['image/encoded']

    decoded = tf.image.decode_jpeg(encoded_image, channels = 3)

    #decoded = tf.Print(decoded, [tf.shape(decoded)])
    if FLAGS.stretch:
        # Stretch to a fixed square
        new_height, new_width = FLAGS.size, FLAGS.size
    else:
        # Preserve aspect ratio and only resize if shorter side > FLAGS.size
        shape = tf.shape(decoded)
        h, w = tf.to_float(shape[0]), tf.to_float(shape[1])
        min_side = tf.minimum(h, w)
        scale = float(FLAGS.size) / min_side
        scale = tf.minimum(scale, 1.0) # Shrink only
        # HACK TESTING upscaling small images to 320
        #dnscale = tf.minimum(float(FLAGS.size) / min_side, 1.0)
        #upscale = tf.maximum(320. / min_side, 1.0)
        #scale = dnscale * upscale

        new_height = tf.cast(scale * h, tf.int32)
        new_width  = tf.cast(scale * w, tf.int32)
    #decoded = tf.Print(decoded, [new_height, new_width])

    resized_float = tf.image.resize_images(
        images = decoded,
        size = [new_height, new_width],
        method = tf.image.ResizeMethod.BILINEAR,
        align_corners = False)
    #resized_float = tf.Print(resized_float, [tf.reduce_min(resized_float),
    #                                         tf.reduce_max(resized_float)])
    resized_uint8 = tf.cast(resized_float, tf.uint8)

    encoded_resized = tf.image.encode_jpeg(
        resized_uint8,
        format='rgb',
        quality=FLAGS.quality,
        progressive=False,
        optimize_size=True,
        chroma_downsampling=True,
        density_unit='in')
    """
    # HACK TESTING
    #print 'xmin, xmax', example['image/object/bbox/xmin'], example['image/object/bbox/xmin']
    #example['image/object/bbox/xmin'] = tf.Print(example['image/object/bbox/xmin'].values,
    #                                             [example['image/object/bbox/xmin'].values])
    # HACK TESTING
    print '*******', example['image/object/bbox/xmin'].values
    bbox = tf.stack([example['image/object/bbox/%s'%x].values
                     for x in ['ymin', 'xmin', 'ymax', 'xmax']])
    bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
    encoded_resized = tf.Print(encoded_resized,
                               [bbox, example['image/object/bbox/xmin'].values])
    """
    return [encoded_resized,
            example['image/height'],
            example['image/width'],
            example['image/channels'],
            example['image/colorspace'],
            example['image/class/label'],
            example['image/class/text'],
            example['image/class/synset'],
            example['image/format'],
            example['image/filename'],
            example['image/object/bbox/xmin'],
            example['image/object/bbox/ymin'],
            example['image/object/bbox/xmax'],
            example['image/object/bbox/ymax']]

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

if __name__ == "__main__":
    import argparse
    import glob
    import sys
    global FLAGS
    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('-i', '--input_dir', required=True)
    cmdline.add_argument('-o', '--output_dir', required=True)
    cmdline.add_argument('-f', '--force', action="store_true")
    cmdline.add_argument('-s', '--subset_name', default='train')
    cmdline.add_argument('-R', '--stretch', action="store_true")
    cmdline.add_argument('-d', '--dataset_name', default=None)
    cmdline.add_argument('-r', '--size', default=480, type=int)
    cmdline.add_argument('-Q', '--quality', default=85, type=int)
    cmdline.add_argument('--start_offset', default=0, type=int)
    cmdline.add_argument('--num_preprocess_threads', default=0, type=int,
                         help="""Number of preprocessing threads.""")
    cmdline.add_argument('--num_intra_threads', default=0, type=int,
                         help="""Number of threads to use for intra-op
                         parallelism. If set to 0, the system will pick
                         an appropriate number.""")
    cmdline.add_argument('--num_inter_threads', default=0, type=int,
                         help="""Number of threads to use for inter-op
                         parallelism. If set to 0, the system will pick
                         an appropriate number.""")
    FLAGS, unknown_args = cmdline.parse_known_args()

    if not FLAGS.num_preprocess_threads:
        FLAGS.num_preprocess_threads = cpu_count()

    if FLAGS.dataset_name is None:
        if   "imagenet" in FLAGS.input_dir: FLAGS.dataset_name = "imagenet"
        elif "flowers"  in FLAGS.input_dir: FLAGS.dataset_name = "flowers"
        else: raise ValueError("Could not identify name of dataset. Please specify with --data_name option.")
    if   FLAGS.dataset_name == "imagenet": dataset = ImagenetData(FLAGS.input_dir)
    elif FLAGS.dataset_name == "flowers":  dataset =  FlowersData(FLAGS.input_dir)
    else: raise ValueError("Unknown dataset. Must be one of imagenet or flowers.")

    infiles = dataset.data_files(FLAGS.subset_name)
    num_shards = len(infiles)
    infiles = infiles[FLAGS.start_offset:]
    num_examples = dataset.num_examples_per_epoch(FLAGS.subset_name)
    examples_per_shard = (num_examples-1) // num_shards + 1

    print(" num_preprocess_threads : {}\n examples_per_shard is {}\n "
          "num_intra_threads is {}\n num_inter_threads is {}".format(FLAGS.num_preprocess_threads, examples_per_shard,
                                                                     FLAGS.num_inter_threads, FLAGS.num_intra_threads))

    config = tf.ConfigProto(
        inter_op_parallelism_threads = FLAGS.num_inter_threads,
        intra_op_parallelism_threads = FLAGS.num_intra_threads)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    filename_queue = tf.train.string_input_producer(
        string_tensor = infiles,
        shuffle = False,
        capacity = examples_per_shard * FLAGS.num_preprocess_threads,
        shared_name = 'filename_queue',
        name = 'filename_queue',
        num_epochs = 1)
    reader = tf.TFRecordReader()
    _, read_op = reader.read(filename_queue)
    examples_queue = tf.FIFOQueue(
        capacity = 2 * examples_per_shard * FLAGS.num_preprocess_threads,
        dtypes=[tf.string])
    example_enqueue_op = examples_queue.enqueue(read_op)
    tf.train.queue_runner.add_queue_runner(
        tf.train.QueueRunner(examples_queue, [example_enqueue_op]))
    example_dequeue_op = examples_queue.dequeue()

    resized_batch = resize_example(example_dequeue_op)
    """
    resized_example_ops = []
    #output_queue = tf.FIFOQueue(
    #    capacity=2*examples_per_shard * FLAGS.num_preprocess_threads,
    #    dtypes=[tf.string])
    #output_enqueue_ops = []
    for t in xrange(FLAGS.num_preprocess_threads):
        output = resize_example(example_dequeue_op)
        resized_example_ops.append(output)
        #output_enqueue_ops.append(output_queue.enqueue(output))
    #output_qr = tf.train.QueueRunner(output_queue, [output_enqueue_op])
    #output_dequeue_op = output_queue.dequeue()
    resized_batch = tf.train.batch_join(
        resized_example_ops,
        batch_size = examples_per_shard,
        capacity = 3 * examples_per_shard)
    print resized_batch
    """
    print("Initializing")
    #init = tf.initialize_local_variables()
    init = tf.local_variables_initializer()
    sess.run(init)

    coordinator = tf.train.Coordinator()
    queue_threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    print("Running")

    batch_num = FLAGS.start_offset
    while not coordinator.should_stop():
        batch_num += 1
        print(batch_num)

        output_filename = '%s-%05d-of-%05d' % (FLAGS.subset_name, batch_num, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        if not os.path.exists(FLAGS.output_dir):
            os.mkdir(FLAGS.output_dir)
        if os.path.exists(output_file) and not FLAGS.force:
            raise IOError("Output file already exists (pass -f to overwrite): " + output_file)
        with tf.python_io.TFRecordWriter(output_file) as writer:
            for i in range(examples_per_shard):

        #print sess.run([t.op for t in resized_batch])
                encoded_images, heights, widths, channels, colorspaces, \
                labels, texts, synsets, img_format, img_filename, \
                    xmin, ymin, xmax, ymax = \
                    sess.run(resized_batch)

        #output_filename = '%s-%05d-of-%05d' % (FLAGS.subset_name, batch_num, num_shards)
        #output_file = os.path.join(FLAGS.output_dir, output_filename)
        #with tf.python_io.TFRecordWriter(output_file) as writer:
            #for rec in xrange(len(encoded_images)):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded':          bytes_feature(encoded_images),
                    'image/height':           int64_feature(heights[0]),
                    'image/width':            int64_feature(widths[0]),
                    'image/channels':         int64_feature(channels[0]),
                    'image/colorspace':       bytes_feature(colorspaces),
                    'image/class/label':      int64_feature(labels[0]),
                    'image/class/text':       bytes_feature(texts),
                    'image/class/synset':     bytes_feature(synsets),
                    'image/format':           bytes_feature(img_format),
                    'image/filename':         bytes_feature(img_filename),
                    'image/object/bbox/xmin': float_feature(xmin.values.tolist()),
                    'image/object/bbox/ymin': float_feature(ymin.values.tolist()),
                    'image/object/bbox/xmax': float_feature(xmax.values.tolist()),
                    'image/object/bbox/ymax': float_feature(ymax.values.tolist()) }))
                writer.write(example.SerializeToString())

    coordinator.request_stop()
    coordinator.join(queue_threads, stop_grace_period_secs=5.)
    sess.close()
