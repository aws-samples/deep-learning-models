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
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 weight_decay=1e-4,
                 interpolation_method='bilinear',
                 use_bias=True):
        """
        in_channels (List[Tuple[str, int]]): list of channel name and number of input channels per level
        out_channels (int): number of output channels at each level
        num_outs (int): number of output scales
        start_level (int): index of first input level to be used, i.e. index that maps to start in in_channels list
        end_level (int): index of last input level to be used, i.e. we have inputs from [start_index:end_index+1]
        add_extra_convs (boolean): often for models like RetinaNet extra conv layers are added per level
        """
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= self.num_ins
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = []
        self.fpn_convs = []
        self.additions = []

        for i in range(self.start_level, self.backbone_end_level):
            channel_name = self.in_channels[i][0]
            lat_kernel_size = 1
            l_conv = layers.Conv2D(out_channels, lat_kernel_size, strides=1, use_bias=use_bias,
                                    name='lateral_{}'.format(channel_name),
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                    padding='same', kernel_initializer='glorot_uniform')
            fpn_kernel_size = 3
            fpn_conv = layers.Conv2D(out_channels, fpn_kernel_size, 1, use_bias=use_bias,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       padding='same', name='fpn_{}'.format(channel_name),
                                       kernel_initializer='glorot_uniform')
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            if i > self.start_level:
                self.additions.append(layers.Add(name='add_{}'.format(channel_name)))

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                extra_fpn_conv = layers.Conv2D(out_channels, (3, 3), strides=2,
                                                padding='same',
                                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                name='extra_conv_{}'.format(i),
                                                kernel_initializer='glorot_uniform')
                self.fpn_convs.append(extra_fpn_conv)
        self._method = interpolation_method


    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):
        # e.g. inputs = (C2, C3, C4, C5)
        assert len(inputs) == len(self.in_channels)
        
        laterals = [lateral_conv(inputs[i + self.start_level])
                        for i, lateral_conv in enumerate(self.lateral_convs)]

        # top down
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels-1, 0, -1):
            feat_shape = tf.shape(laterals[i-1]) # NHWC
            h, w = feat_shape[1], feat_shape[2]
            channel_name = self.in_channels[i][0]
            upsampled = tf.image.resize(laterals[i], (h, w), method=self._method, name='{}_resize'.format(channel_name))
            laterals[i-1] = layers.Add()([laterals[i-1], upsampled])

        # outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]        

        # extra levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(tf.nn.max_pool(outs[i - 1], ksize=1, strides=2, padding='SAME', name='build_p{}'.format(used_backbone_levels - i)))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    outs.append(self.fpn_convs[i](outs[-1]))
        return outs

