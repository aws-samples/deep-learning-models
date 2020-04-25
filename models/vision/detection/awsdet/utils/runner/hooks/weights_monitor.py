# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from .hook import Hook
from ..dist_utils import master_only
import re

def print_weights(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        conv_names = ['fpn_c5p5','conv4_block1_1_conv','conv3_block2_3_conv','rpn_class_raw', 'rpn_bbox_pred']
        for p in conv_names:
            if re.search(p, name):
                print(name)
                print(weight[0])
        bn_names = ['conv4_block1_1_bn', 'conv3_block1_1_bn']
        for p in bn_names:
            if re.search(p, name):
                print(name)
                print(weight[0])


class WeightsMonitorHook(Hook):

    @master_only
    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if cur_iter % 1000 != 0:
            return
        print_weights(runner.model)

