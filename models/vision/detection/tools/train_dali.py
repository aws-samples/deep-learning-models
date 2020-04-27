# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tarfile
import os
import pathlib
import numpy as np
from time import time
# temp
import sys
sys.path.append('.')

import tensorflow_addons as tfa
import tensorflow as tf
import horovod.tensorflow as hvd

from awsdet.datasets import DATASETS, build_dataloader, build_dataset, dali
from awsdet.models import build_detector
from awsdet.utils.schedulers import schedulers
from awsdet.core import CocoDistEvalmAPHook, CocoDistEvalRecallHook
from awsdet.utils.runner.hooks.logger import tensorboard, text
from awsdet.utils.runner.hooks import checkpoint, iter_timer, visualizer
from awsdet.apis.train import parse_losses, batch_processor, build_optimizer, get_root_logger
from awsdet.utils.misc import Config
import horovod.tensorflow as hvd
from awsdet.utils.runner import sagemaker_runner
import argparse

##########################################################################################
# Setup horovod and tensorflow environment
##########################################################################################

fp16 = True
hvd.init()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": fp16})
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

def main(cfg):
    ######################################################################################
    # Create Training Data
    ######################################################################################
    datasets = build_dataset(cfg.data.train)
    tf_datasets = [build_dataloader(datasets,
                         cfg.batch_size_per_device,
                         cfg.workers_per_gpu,
                         num_gpus=hvd.size(),
                         dist=True)]
    dali_tdf = dali.dali_dataset(cfg.data.train.dataset_dir, 
                                 batch_size=cfg.batch_size_per_device, 
                                 device_id=hvd.rank(), num_gpus=hvd.size())
    tf_datasets = [(dali_tdf, tf_datasets[0][1])]
    ######################################################################################
    # Build Model
    ######################################################################################
    model = build_detector(cfg.model,
                           train_cfg=cfg.train_cfg,
                           test_cfg=cfg.test_cfg)
    # Pass example through so tensor shapes are defined
    _ = model(next(iter(tf_datasets[0][0])), use_dali=True)
    model.layers[0].layers[0].load_weights(cfg.weights_path, by_name=False)
    ######################################################################################
    # Create Model Runner
    ######################################################################################
    runner = sagemaker_runner.Runner(model, batch_processor, name=cfg.model_name, 
                                     optimizer=cfg.optimizer, work_dir=cfg.work_dir,
                                     logger=get_root_logger(cfg.log_level), amp_enabled=cfg.fp16,
                                     loss_weights=cfg.loss_weights, use_dali=True)
    runner.timestamp = int(time())
    ######################################################################################
    # Setup Training Hooks
    ######################################################################################
    runner.register_hook(checkpoint.CheckpointHook(interval=cfg.checkpoint_interval, 
                                                   out_dir=cfg.outputs_path, 
                                                   s3_dir=None))
    runner.register_hook(CocoDistEvalmAPHook(cfg.data.val, interval=cfg.evaluation_interval))
    runner.register_hook(iter_timer.IterTimerHook())
    runner.register_hook(text.TextLoggerHook())
    runner.register_hook(visualizer.Visualizer(cfg.data.val, interval=100, top_k=10))
    runner.register_hook(tensorboard.TensorboardLoggerHook(log_dir=cfg.outputs_path, 
                                                           image_interval=100, s3_dir=None))
    ######################################################################################
    # Run Model
    ######################################################################################
    runner.run(tf_datasets, cfg.workflow, cfg.training_epochs)

def parse():
    """
    Parse path to configuration file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="Model configuration file")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse()
    cfg = Config.fromfile(args.configuration)
    cfg.model_name = "demo"
    main(cfg)