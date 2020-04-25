# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
'''
FPN model for Keras.

# Reference:
- [Feature Pyramid Networks for Object Detection](
    https://arxiv.org/abs/1612.03144)

'''
import tensorflow as tf
from tensorflow.keras import layers
from ..registry import NECKS

@NECKS.register_module
class FPN(tf.keras.Model):
    def __init__(self, top_down_dims=256, weight_decay=1e-4, interpolation_method='bilinear', use_bias=True):
        super().__init__()

        self._build_p5_conv = layers.Conv2D(top_down_dims, 1, strides=1, use_bias=use_bias, name='build_p5',
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                            kernel_initializer='he_normal')
        self._build_p6_max_pooling = layers.MaxPooling2D(strides=2, pool_size=(1, 1), name='build_p6')

        self._build_p4_reduce_dims = layers.Conv2D(top_down_dims, 1, strides=1, use_bias=use_bias,
                                                   name='build_p4_reduce_dims',
                                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                   kernel_initializer='he_normal')
        self._build_p4_fusion = layers.Add(name='build_p4_fusion')
        self._build_p4 = layers.Conv2D(top_down_dims, 3, 1, use_bias=use_bias,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       padding='same', name='build_p4',
                                       kernel_initializer='he_normal')

        self._build_p3_reduce_dims = layers.Conv2D(top_down_dims, 1, strides=1, use_bias=use_bias,
                                                   name='build_p3_reduce_dims',
                                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                   kernel_initializer='he_normal')
        self._build_p3_fusion = layers.Add(name='build_p3_fusion')
        self._build_p3 = layers.Conv2D(top_down_dims, 3, 1, use_bias=use_bias,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       padding='same', name='build_p3',
                                       kernel_initializer='he_normal')

        self._build_p2_reduce_dims = layers.Conv2D(top_down_dims, 1, strides=1, use_bias=use_bias,
                                                   name='build_p2_reduce_dims',
                                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                   kernel_initializer='he_normal')
        self._build_p2_fusion = layers.Add(name='build_p2_fusion')
        self._build_p2 = layers.Conv2D(top_down_dims, 3, 1, use_bias=use_bias,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       padding='same', name='build_p2',
                                       kernel_initializer='he_normal')
        self._method = interpolation_method


    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):
        c2, c3, c4, c5 = inputs

        # build p5 & p6
        p5 = self._build_p5_conv(c5)
        p6 = self._build_p6_max_pooling(p5)

        # build p4
        h, w = tf.shape(c4)[1], tf.shape(c4)[2]
        upsample_p5 = tf.image.resize(p5, (h, w), method=self._method, name='build_p4_resize')
        reduce_dims_c4 = self._build_p4_reduce_dims(c4)
        p4 = self._build_p4_fusion([upsample_p5 * 0.5, reduce_dims_c4 * 0.5])

        # build p3
        h, w = tf.shape(c3)[1], tf.shape(c3)[2]
        upsample_p4 = tf.image.resize(p4, (h, w), method=self._method, name='build_p3_resize')
        reduce_dims_c3 = self._build_p3_reduce_dims(c3)
        p3 = self._build_p3_fusion([upsample_p4 * 0.5, reduce_dims_c3 * 0.5])

        # build p2
        h, w = tf.shape(c2)[1], tf.shape(c2)[2]
        upsample_p3 = tf.image.resize(p3, (h, w), method=self._method, name='build_p2_resize')
        reduce_dims_c2 = self._build_p2_reduce_dims(c2)
        p2 = self._build_p2_fusion([upsample_p3 * 0.5, reduce_dims_c2 * 0.5])

        p4 = self._build_p4(p4)
        p3 = self._build_p3(p3)
        p2 = self._build_p2(p2)

        return p2, p3, p4, p5, p6
