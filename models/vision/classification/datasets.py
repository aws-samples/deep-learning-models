import horovod.tensorflow as hvd
import os
import tensorflow as tf
from preprocessing import resnet_preprocessing, imagenet_preprocessing, darknet_preprocessing
import functools

def create_dataset(data_dir, batch_size, preprocessing='resnet', validation=False):
    filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
    data = tf.data.TFRecordDataset(filenames).shard(hvd.size(), hvd.rank())
    if not validation:
        parse_fn = functools.partial(parse_train, preprocessing=preprocessing)
        data = data.shuffle(buffer_size=1000)
        data = data.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        parse_fn = functools.partial(parse_validation, preprocessing=preprocessing)
        data = data.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # we drop remainder because we want same sized batches - XLA and because of allreduce being used to calculate
    # accuracy - validation accuracy may be slightly different than computing on all of validation data
    data = data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return data


@tf.function
def parse(record, is_training, preprocessing): 
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
    if preprocessing == 'resnet':
        augmenter = None # augment.AutoAugment()
        image = resnet_preprocessing.preprocess_image(image_bytes, bbox, 224, 224, 3, is_training=is_training)
    elif preprocessing == 'imagenet': # used by hrnet
        image = imagenet_preprocessing.preprocess_image(image_bytes, bbox, 224, 224, 3, is_training=is_training)
    elif preprocessing == 'darknet':
        image = darknet_preprocessing.preprocess_image(image_bytes, bbox, 256, 256, 3, is_training=is_training)


    label = tf.cast(parsed['image/class/label'] - 1, tf.int32)
    one_hot_label = tf.one_hot(label, depth=1000, dtype=tf.float32)
    return image, one_hot_label


def parse_train(record, preprocessing):
    return parse(record, is_training=True, preprocessing=preprocessing)


def parse_validation(record, preprocessing):
    return parse(record, is_training=False, preprocessing=preprocessing)

