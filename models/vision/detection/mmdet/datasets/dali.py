# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import pathlib
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from time import time
import nvidia.dali.plugin.tf as dali_tf

class COCOPipeline(Pipeline):
    
    def __init__(self, file_root, annotations_file,
                 batch_size, num_threads, device_id=0, num_gpus=1, 
                 mean=(123.675, 116.28, 103.53), stddev=(1., 1., 1.),
                 random_shuffle=True):
        super(COCOPipeline, self).__init__(batch_size, num_threads, device_id, seed = 15)
        self.input = ops.COCOReader(file_root=file_root, annotations_file=annotations_file,
                                    shard_id=device_id, num_shards=num_gpus, ratio=True, 
                                    skip_empty=True, prefetch_queue_depth=32, random_shuffle=True)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.BGR)
        self.resize = ops.Resize(device='gpu', max_size=1216, resize_shorter=800)
        self.flip = ops.CoinFlip(device='cpu')
        self.bbox_flip = ops.BbFlip(device='gpu')
        self.CMN = ops.CropMirrorNormalize(device='gpu', mean=mean, std=stddev, output_layout='HWC')
        self.image_pad = ops.Pad(device='gpu', fill_value=0, axes=(0,1), shape=(1216, 1216))
        self.bbox_pad = ops.Pad(device='gpu', fill_value=0, axes=(0,), shape=(100,))
        self.label_pad = ops.Pad(device='gpu', fill_value=-1, axes=(0,), shape=(100,))
        self.get_shape = ops.Shapes(device='gpu')
        self.float_cast = ops.Cast(device='gpu', dtype=types.FLOAT)

    
    def define_graph(self):
        inputs, bboxes, labels = self.input()
        images = self.decode(inputs)
        orig_image_shape = self.get_shape(images)
        images = self.resize(images)
        flip = self.flip().gpu()
        images = self.CMN(images, mirror=flip)
        bboxes = self.bbox_flip(bboxes.gpu(), horizontal=flip)
        resized_image_shape = self.get_shape(images)
        images = self.image_pad(images)
        padding_shape = self.get_shape(images)
        bboxes = self.bbox_pad(bboxes)
        labels = self.label_pad(labels.gpu())
        orig_image_shape = self.float_cast(orig_image_shape)
        resized_image_shape = self.float_cast(resized_image_shape)
        padding_shape = self.float_cast(padding_shape)
        flip = self.float_cast(flip)
        return (images, bboxes, labels, 
                orig_image_shape, 
                resized_image_shape,
                padding_shape,
                flip)
    
def dali_dataset(data_dir, batch_size=1, image_size=1216, device_id=0, num_gpus=1, subset='train', year='2017'):
    images = pathlib.Path(data_dir).joinpath('{}{}'.format(subset, year)).as_posix()
    annotations = pathlib.Path(data_dir).joinpath('annotations').joinpath('instances_{}{}.json'.format(subset, year)).as_posix()
    pipe = COCOPipeline(file_root=images, 
                        annotations_file=annotations,
                        batch_size=batch_size, 
                        num_threads=4,
                        device_id=device_id,
                        num_gpus=num_gpus)
    shapes = [
        (batch_size, 1216, 1216, 3),
        (batch_size, 100, 4),
        (batch_size, 100),
        (batch_size, 3),
        (batch_size, 3),
        (batch_size, 3),
        (batch_size)]
    
    dtypes = [
        tf.float32,
        tf.float32,
        tf.int32,
        tf.float32,
        tf.float32,
        tf.float32,
        tf.float32]
    
    tf_pipe = dali_tf.DALIDataset(
        pipeline=pipe,
        batch_size=batch_size,
        shapes=shapes,
        dtypes=dtypes,
        device_id=device_id)
    
    return tf_pipe

def dali_adapter(img, bboxes, labels, original_size, resized_size, padded_size, flip, training=True):
    scale_factor = tf.expand_dims(resized_size[...,0]/original_size[...,0], axis=1)
    flip = tf.expand_dims(flip, axis=1)
    meta = tf.concat([original_size, resized_size, padded_size, scale_factor, flip], axis=1)
    H = tf.expand_dims(resized_size[...,0], axis=1)
    W = tf.expand_dims(resized_size[...,1], axis=1)
    y1 = tf.expand_dims(bboxes[...,1]*H, axis=2)
    x1 = tf.expand_dims(bboxes[...,0]*W, axis=2)
    y2 = tf.expand_dims(bboxes[...,1]*H + bboxes[...,3]*H, axis=2)
    x2 = tf.expand_dims(bboxes[...,0]*W + bboxes[...,2]*W, axis=2)
    bboxes = tf.concat([y1, x1, y2, x2], axis=2)
    
    if training:
        return img, meta, bboxes, labels
    return img, meta