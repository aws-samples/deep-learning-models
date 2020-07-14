# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf

from .. import data_generator
from awsdet.utils.runner.dist_utils import get_dist_info


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu=1, # unused
                     num_gpus=0,
                     dist=True,
                     shuffle=True,
                     **kwargs):
    """Build a TF Dataset pipeline that returns padded batches.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU. - TODO: unused
        num_gpus (int): Number of GPUs. Only used in non-distributed training - TODO
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        tf.data.Dataset: A TF dataset pipeline.
    """
    batch_size = imgs_per_gpu
    if dist:
        rank, local_rank, size, local_size = get_dist_info()
        if dataset.train:
            generator = data_generator.DataGenerator(dataset, index=rank, num_gpus=size, shuffle=shuffle)
        else:
            generator = data_generator.DataGenerator(dataset, index=rank, num_gpus=local_size, shuffle=False) # evaluation on node 0 workers
    else:
        generator = data_generator.DataGenerator(dataset, shuffle=False)

    if dataset.train:
        if dataset.mask:
            tf_dataset = tf.data.Dataset.from_generator(
                generator, (tf.float32, tf.float32, tf.float32, tf.int32, tf.int32))
        else:
            tf_dataset = tf.data.Dataset.from_generator(
                generator, (tf.float32, tf.float32, tf.float32, tf.int32))
        
        tf_dataset = tf_dataset.map(lambda *args: args, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        if dataset.mask:
            tf_dataset = tf_dataset.padded_batch(
                                batch_size,
                                padded_shapes=(
                                tf.TensorShape([None, None, 3]), # image padded to largest in batch
                                tf.TensorShape([11]),            # image meta - no padding
                                tf.TensorShape([None, 4]),       # bounding boxes, padded to longest
                                tf.TensorShape([None]),          # labels, padded to longest
                                tf.TensorShape([None, None, None]) # pad masks to largest in batch
                                ),
                                padding_values=(0.0, 0.0, 0.0, -1, -1))
        
        else:
            tf_dataset = tf_dataset.padded_batch(
                                batch_size,
                                padded_shapes=(
                                tf.TensorShape([None, None, 3]), # image padded to largest in batch
                                tf.TensorShape([11]),            # image meta - no padding
                                tf.TensorShape([None, 4]),       # bounding boxes, padded to longest
                                tf.TensorShape([None])           # labels, padded to longest
                                ),
                                padding_values=(0.0, 0.0, 0.0, -1))

        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return tf_dataset, generator.num_examples // batch_size
    else:
        tf_dataset = tf.data.Dataset.from_generator(
            generator, (tf.float32, tf.float32))
        tf_dataset = tf_dataset.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None, None,
                                3]),  # image padded to largest in batch
                tf.TensorShape([11]),  # image meta - no padding
            ),
            padding_values=(0.0, 0.0))
        tf_dataset = tf_dataset.repeat()
        return tf_dataset, generator.num_examples // batch_size

