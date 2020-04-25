# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import functools
import os
import subprocess
import tensorflow as tf
import horovod.tensorflow as hvd

def init_dist():
    hvd.init()
    # tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

def get_dist_info():
    return hvd.rank(), hvd.local_rank(), hvd.size(), hvd.local_size() #TODO return a dict instead

def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _, _, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def broadcast_weights(runner):
    print('Rank {} broadcasting'.format(runner.rank))
    hvd.broadcast_variables(runner.model.variables, root_rank=0)
    hvd.broadcast_variables(runner.optimizer.variables(), root_rank=0)
    print('Variable broadcast done.')

def get_distributed_tape(tape):
    return hvd.DistributedGradientTape(tape)

def get_barrier():
    return hvd.allreduce(tf.constant(0, dtype=tf.float32))
