# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import re
import tensorflow as tf

def flatten_list(nested, result):
    for n in nested:
        if isinstance(n, list):
            flatten_list(n, result)
        else:
            result.append(n)


def get_sublayers(m):
    result = []
    if hasattr(m, 'layers'): # layer subclasses keras.Model
        flatten_list(m.layers, result)
    if hasattr(m, '_layers') and len(m._layers) > 0: # layer does not subclass keras.Model
        flatten_list(m._layers, result)
    return result


def freeze_model_layers(model, patterns_list):
    def freeze_nested(l, level=1):
        frozen = False
        for p in patterns_list:
            if isinstance(l, tf.keras.layers.Layer) and re.search(p, l.name):
                print('Freezing (trainable: False)', l.name)
                l.trainable = False
                frozen = True
                break
        if not frozen:
            for ll in get_sublayers(l):
                freeze_nested(ll, level + 1)

    for l in get_sublayers(model):
        freeze_nested(l)

