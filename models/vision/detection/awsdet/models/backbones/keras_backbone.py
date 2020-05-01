# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from ..registry import BACKBONES
from awsdet.utils.keras import get_base_model, get_outputs


@BACKBONES.register_module
class KerasBackbone(tf.keras.Model):
    def __init__(self, model_name, weights_path=None, weight_decay=1e-4, **kwargs):
        super(KerasBackbone, self).__init__(**kwargs)
        self.model_name = model_name
        self.weights_path = weights_path
        _base_model = get_base_model(model_name, weights_path, weight_decay=weight_decay)
        self._model = tf.keras.Model(inputs=_base_model.input,
                                     outputs=get_outputs(_base_model), name=model_name)


    @tf.function
    def call(self, inputs, training=True):
        c2, c3, c4, c5 = self._model(inputs, training=False) # freeze BN
        if self.model_name == 'ResNet50V2':
            # resnet 50 v2 returns half the size of feature maps that affects FPN output
            # for now we resize the feature maps but should handle generically in FPN in the future
            c2_shape = tf.multiply(2, tf.shape(c2)[1:3])
            c3_shape = tf.shape(c2)[1:3]
            c4_shape = tf.shape(c3)[1:3]
            c2 = tf.image.resize(c2, c2_shape, method='bilinear', name='c2_resize')
            c3 = tf.image.resize(c3, c3_shape, method='bilinear', name='c3_resize')
            c4 = tf.image.resize(c4, c4_shape, method='bilinear', name='c4_resize')
        return (c2, c3, c4, c5)
