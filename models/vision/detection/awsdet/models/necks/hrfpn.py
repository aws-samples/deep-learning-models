# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

'''
FPN model for HRNet backbone.

# Reference:
- [Feature Pyramid Networks for Object Detection](
    https://arxiv.org/abs/1612.03144)

'''
import tensorflow as tf
from tensorflow.keras import layers
from ..registry import NECKS

@NECKS.register_module
class HRFPN(tf.keras.Model):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 weight_decay=1e-4,
                 use_bias=True):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.reduction_conv = layers.Conv2D(out_channels, 1, use_bias=use_bias,
                name='reduction_conv',
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                padding='same', kernel_initializer='he_uniform')

        self.fpn_convs = []
        for i in range(self.num_outs):
            fpn_kernel_size = 3
            fpn_conv = layers.Conv2D(out_channels, fpn_kernel_size, 1, use_bias=use_bias,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                padding='same', name='fpn_out{}'.format(i),
                kernel_initializer='he_uniform')
            self.fpn_convs.append(fpn_conv)


    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        # e.g. inputs = (C2, C3, C4, C5)
        assert len(inputs) == len(self.in_channels)
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            feat_shape = tf.shape(inputs[i])
            scale_factor = 2**i
            h = feat_shape[1] * scale_factor
            w = feat_shape[2] * scale_factor
            upsampled = tf.image.resize(inputs[i], (h, w), method='bilinear', name='resize_{}'.format(i))
            outs.append(upsampled)
        out = tf.concat(outs, axis=-1)
        out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(tf.nn.avg_pool(out, 2**i, strides=2**i, padding='VALID'))
        outputs = []
        for i in range(self.num_outs):
            outputs.append(self.fpn_convs[i](outs[i]))
        return outputs


