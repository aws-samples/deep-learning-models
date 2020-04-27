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
sys.path.append('..')

import tensorflow_addons as tfa
import tensorflow as tf
import horovod.tensorflow as hvd

from awsdet.datasets import DATASETS, build_dataloader
from awsdet.datasets import build_dataset, build_dataloader
from awsdet.models import build_detector
from awsdet.utils.schedulers import schedulers
from awsdet.core import CocoDistEvalmAPHook, CocoDistEvalRecallHook
from awsdet.utils.runner.hooks.logger import tensorboard, text
from awsdet.utils.runner.hooks import checkpoint, iter_timer, visualizer
from awsdet.apis.train import parse_losses, batch_processor, build_optimizer, get_root_logger, set_random_seed
from awsdet.utils.misc import Config
import horovod.tensorflow as hvd
from awsdet.utils.runner import sagemaker_runner
from awsdet.utils.schedulers.schedulers import WarmupScheduler
import argparse

##########################################################################################
# Setup horovod and tensorflow environment
##########################################################################################

os.environ['TF_CUDNN_USE_AUTOTUNE']= str(0)

fp16 = True
hvd.init()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": fp16})
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

##########################################################################################
# Set seed for RNG
##########################################################################################
set_random_seed(1337 + hvd.rank(), deterministic=True)


##########################################################################################
# Data is downloaded in tar archive to /opt/ml/inputs/data/coco coco channel
# First need to untar the coco data
##########################################################################################

def decompress_data(cfg):
    if hvd.local_rank()==0:
        print("Decompressing Data")
        coco_tar = tarfile.open(pathlib.Path(os.getenv('SM_CHANNEL_COCO')).joinpath('coco.tar').as_posix())
        coco_tar.extractall(path=os.getenv('SM_CHANNEL_COCO'))
    # block other ranks form skipping ahead before data is ready
    barrier = hvd.allreduce(tf.random.normal(shape=[1]))    

def setup_paths(instance_name, s3_path):
    s3_checkpoints = os.path.join(s3_path, "checkpoints", instance_name)
    s3_tensorboard = os.path.join(s3_path, "tensorboard", instance_name)
    return s3_checkpoints, s3_tensorboard
    
def main(cfg):
    decompress_data(cfg)
    ######################################################################################
    # Create Training Data
    ######################################################################################
    cfg.global_batch_size = cfg.batch_size_per_device * hvd.size()
    cfg.steps_per_epoch = cfg.coco_images // cfg.global_batch_size

    datasets = build_dataset(cfg.data.train)
    tf_datasets = [build_dataloader(datasets,
                         cfg.batch_size_per_device,
                         cfg.workers_per_gpu,
                         num_gpus=hvd.size(),
                         dist=True)]
    ######################################################################################
    # Build Model
    ######################################################################################
    
    #update any hyperparams that we may have passed in via arguments
    if cfg.ls > 0.0:
        cfg.model['bbox_head']['label_smoothing'] = cfg.ls
    if cfg.use_rcnn_bn:
        cfg.model['bbox_head']['use_bn'] = cfg.use_rcnn_bn
    if cfg.use_conv:
        cfg.model['bbox_head']['use_conv'] = cfg.use_conv

    cfg.schedule = args.schedule
    model = build_detector(cfg.model,
                           train_cfg=cfg.train_cfg,
                           test_cfg=cfg.test_cfg)
    # Pass example through so tensor shapes are defined
    _ = model(next(iter(tf_datasets[0][0])))
    model.layers[0].layers[0].load_weights(cfg.weights_path, by_name=False)

    ######################################################################################
    # Build optimizer and associate scheduler
    ######################################################################################

    # base learning rate is set for global batch size of 8, with linear scaling for larger batches
    base_learning_rate = cfg.base_learning_rate
    scaled_learning_rate = base_learning_rate * cfg.global_batch_size / 8
    steps_per_epoch = cfg.steps_per_epoch
    if cfg.schedule == '1x':
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [steps_per_epoch * 8, steps_per_epoch * 10],
            [scaled_learning_rate, scaled_learning_rate*0.1, scaled_learning_rate*0.01])
    elif cfg.schedule == 'cosine':
        scheduler = tf.keras.experimental.CosineDecayRestarts(
            initial_learning_rate=scaled_learning_rate,
            first_decay_steps=12*steps_per_epoch, t_mul=1, m_mul=1) #0-1-13
    else:
        raise NotImplementedError
    warmup_init_lr = 1.0 / cfg.warmup_init_lr_scale * scaled_learning_rate
    scheduler = WarmupScheduler(scheduler, warmup_init_lr, cfg.warmup_steps)
    # FIXME: currently hardcoded to SGD
    optimizer = tf.keras.optimizers.SGD(scheduler, momentum=0.9, nesterov=False)
    if cfg.fp16:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scale='dynamic')

    
    ######################################################################################
    # Create Model Runner
    ######################################################################################
    runner = sagemaker_runner.Runner(model, batch_processor, name=cfg.model_name, 
                                     optimizer=optimizer, work_dir=cfg.work_dir,
                                     logger=get_root_logger(cfg.log_level), amp_enabled=cfg.fp16,
                                     loss_weights=cfg.loss_weights)
    runner.timestamp = int(time())
    ######################################################################################
    # Setup Training Hooks
    ######################################################################################
    runner.register_hook(checkpoint.CheckpointHook(interval=cfg.checkpoint_interval, 
                                                   out_dir=cfg.outputs_path, 
                                                   s3_dir=cfg.s3_checkpoints,
                                                   h5=True))
    runner.register_hook(CocoDistEvalmAPHook(cfg.data.val, interval=cfg.evaluation_interval))
    runner.register_hook(iter_timer.IterTimerHook())
    runner.register_hook(text.TextLoggerHook())
    runner.register_hook(visualizer.Visualizer(cfg.data.val, interval=100, top_k=10))
    runner.register_hook(tensorboard.TensorboardLoggerHook(log_dir=cfg.outputs_path, 
                                                           image_interval=100,
                                                           s3_dir=cfg.s3_tensorboard))
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
    parser.add_argument("--instance_name", help="Sagemaker instance name")
    parser.add_argument("--instance_count", help="Number of instances")
    parser.add_argument("--instance_type", help="Instance type for a worker")
    parser.add_argument("--num_workers_per_host", help="Number of workers on each instance")
    parser.add_argument("--s3_path", help="s3 path")
    parser.add_argument("--model_dir", help="Location of model on Sagemaker instance")
    parser.add_argument("--base_learning_rate", help="float")
    parser.add_argument("--batch_size_per_device", help="integer")
    parser.add_argument("--fp16", help="boolean")
    parser.add_argument("--schedule", help="learning rate schedule type")
    parser.add_argument("--warmup_init_lr_scale", help="float")
    parser.add_argument("--warmup_steps", help="int")
    parser.add_argument("--use_rcnn_bn", help="bool")
    parser.add_argument("--use_conv", help="bool")
    parser.add_argument("--ls", help="float")

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse()
    cfg = Config.fromfile(args.configuration)
    instance_name = args.instance_name
    s3_path = args.s3_path
    s3_checkpoints, s3_tensorboard = setup_paths(instance_name, s3_path)
    cfg.s3_checkpoints = s3_checkpoints
    cfg.s3_tensorboard = s3_tensorboard
    cfg.model_name = instance_name
    cfg.base_learning_rate = float(args.base_learning_rate)
    cfg.instance_count = int(args.instance_count)
    cfg.batch_size_per_device = int(args.batch_size_per_device)
    cfg.fp16 = (args.fp16 == 'True')
    cfg.ls = float(args.ls)
    cfg.use_rcnn_bn = (args.use_rcnn_bn == 'True')
    cfg.use_conv = (args.use_conv == 'True')           
    cfg.schedule = args.schedule
    cfg.num_workers_per_host = int(args.num_workers_per_host)
    cfg.workers_per_gpu = 1 # unused
    cfg.warmup_init_lr_scale = float(args.warmup_init_lr_scale)
    cfg.warmup_steps = int(args.warmup_steps)
    main(cfg)
