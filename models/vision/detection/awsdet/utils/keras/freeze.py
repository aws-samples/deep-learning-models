# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import re

def freeze_model_layers(model, patterns_list):
    def freeze_nested(l, level=1):
        frozen = False
        for p in patterns_list:
            print(p, l.name)
            if re.search(p, l.name):
                print('Freezing (trainable: False)', l.name)
                l.trainable = False
                frozen = True
                break
        if not frozen:
            print('NOT freezing (trainable: True)', l.name)
        if hasattr(l, 'layers'):
            for ll in l.layers:
                freeze_nested(ll, level + 1)

    for l in model.layers:
        freeze_nested(l)
