# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from awsdet.core.mask.mask_target import MaskTarget
from awsdet.models.losses.losses import rcnn_mask_loss
from ..registry import HEADS

@HEADS.register_module
class MaskHead(tf.keras.Model):
    def __init__(self, num_classes,
                       weight_decay=1e-5, 
                       use_bn=False,
                       hidden_dim=256,
                       depth=4,
                       max_fg=128,
                       num_rois=512,
                       mask_size=(28, 28),
                       loss=rcnn_mask_loss):
        super().__init__()
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.use_bn = use_bn
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.mask_size = mask_size
        self.mask_target = MaskTarget(max_fg, num_rois, mask_size)
        self.loss = loss
        self.max_fg = tf.concat([tf.ones(max_fg), tf.zeros(num_rois-max_fg)], axis=0)
        self._convs = [tf.keras.layers.Conv2D(self.hidden_dim, (3, 3),
                                              padding="same",
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, 
                                                                                                       mode='fan_out'),
                                              kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                              name="mask_conv_{}".format(i)) for i in range(self.depth)]
        self._bns = [tf.keras.layers.BatchNormalization(name="mask_bn_{}".format(i)) for i in range(self.depth)]
        self._deconv = tf.keras.layers.Conv2DTranspose(self.hidden_dim, (2, 2), strides=2,
                                                       activation=tf.keras.activations.relu,
                                                       kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, 
                                                                                                       mode='fan_out'),
                                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                       name="mask_deconv")
        self._masks = tf.keras.layers.Conv2D(self.num_classes, (1, 1),
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.001),
                                             kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                             strides=1, name="mask_output")

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True):
        x = tf.concat(inputs, axis=0)
        for layer in range(self.depth):
            x = self._convs[layer](x)
            if self.use_bn:
                x = self._bns[layer](x, training=training)
            x = tf.keras.activations.relu(x)
        x = self._deconv(x)
        x = self._masks(x)
        return x

    def get_fg_rois_list(self, rois_list):
        return [tf.boolean_mask(i, self.max_fg) for i in rois_list]

    @tf.function(experimental_relax_shapes=True)
    def mold_masks(self, masks, bboxes, img_meta, threshold=0.5):
        mask_array = tf.TensorArray(tf.int32, size=tf.shape(masks)[0])
        bboxes = tf.cast(bboxes, tf.int32)
        img_meta = tf.cast(img_meta, tf.int32)
        for idx in tf.range(100):
            mask_array = mask_array.write(idx, self._mold_single_mask(masks[idx], bboxes[idx], img_meta, threshold))
        mask_array = mask_array.stack()
        return mask_array

    @tf.function(experimental_relax_shapes=True)
    def _mold_single_mask(self, mask, bbox, img_meta, threshold=0.5):
        '''
        Resize a mask and paste to background for image
        '''
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2] 
        x2 = bbox[3]
        h = y2 - y1
        w = x2 - x1
        if tf.math.multiply(h, w)<=0:
            return tf.zeros((img_meta[6], img_meta[7], 1), dtype=tf.int32)
        mask = tf.math.sigmoid(mask)
        mask_resize = tf.cast(tf.image.resize(mask, (h, w))>threshold, tf.int32)
        pad = [[y1, img_meta[6]-y2], [x1, img_meta[7]-x2], [0,0]]
        mask_resize = tf.pad(mask_resize, pad)
        return mask_resize
