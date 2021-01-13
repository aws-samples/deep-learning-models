path = (
    "/Users/nieljare/Desktop/tensorflow_datasets_c4_en_2.3.1_c4-validation.tfrecord-00000-of-00008"
)

import tensorflow as tf

raw_dataset = tf.data.TFRecordDataset(path)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
