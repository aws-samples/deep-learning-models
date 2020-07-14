# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from .hook import Hook
from awsdet.utils.runner.dist_utils import master_only

class Profiler(Hook):
    """
    Runs the tensorboard profiler tool at a set interval
    """
    def __init__(self,
                 log_dir,
                 interval=500,
                 run_steps=50,
                 run_every_epoch=False):
        self.log_dir = log_dir
        self.interval = interval
        self.run_steps = run_steps
        self.run_every_epoch = '_inner_iter' if run_every_epoch else '_iter'
        
    @master_only
    def after_train_iter(self, runner):
        if runner.__dict__[self.run_every_epoch] + 1 == self.interval:
            tf.profiler.experimental.start(self.log_dir)
        elif runner.__dict__[self.run_every_epoch] + 1 == self.interval + self.run_steps:
            tf.profiler.experimental.stop()