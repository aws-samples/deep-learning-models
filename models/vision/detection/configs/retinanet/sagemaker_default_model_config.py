# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
########################################################################################################################
# Example configuration file for training with Sagemaker
# Pass to train_sagemaker.py as
# python train_sagemaker.py configs/sagemaker_configuration.py
########################################################################################################################

import tensorflow as tf
import tensorflow_addons as tfa
import horovod.tensorflow as hvd
import os
import pathlib
from awsdet.utils.schedulers.schedulers import WarmupScheduler
hvd.init()

########################################################################################################################
# Model artifact paths
# Data input and output locations
########################################################################################################################

dataset_type = 'CocoDataset'
data_root = pathlib.Path(os.getenv('SM_CHANNEL_COCO')).joinpath('coco').as_posix()

weights_file = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = pathlib.Path(os.getenv('SM_CHANNEL_WEIGHTS')).joinpath(weights_file).as_posix()

outputs_path = pathlib.Path(os.getenv('SM_OUTPUT_DATA_DIR')).as_posix()
output_weights = pathlib.Path(os.getenv('SM_OUTPUT_DATA_DIR')).joinpath('faster-rcnn-resnet-50-weights').as_posix()

########################################################################################################################
# Hyperparameters
# More common stuff to adjust
########################################################################################################################

training_epochs = 13
log_level = 'INFO'
work_dir = './work_dirs/retinanet_r50_fpn_1x'
load_from = None
resume_from = None
loss_weights = None
workflow = [('train', 1)]
coco_images = 117504

########################################################################################################################
# Checkpoints and logging
########################################################################################################################
checkpoint_interval = 1
evaluation_interval = 1
########################################################################################################################
# Dataset Settings
########################################################################################################################

data = dict(
    imgs_per_gpu=4,
    train=dict(
        type=dataset_type,
        train=True,
        dataset_dir=data_root,
        subset='train',
        flip_ratio=0.5,
        pad_mode='fixed',
        preproc_mode='caffe',
        mean=(123.675, 116.28, 103.53),
        std=(1., 1., 1.),
        scale=(800, 1333)),
    val=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode='caffe',
        mean=(123.675, 116.28, 103.53),
        std=(1., 1., 1.),
        scale=(800, 1333)),
    test=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode='caffe',
        mean=(123.675, 116.28, 103.53),
        std=(1., 1., 1.),
        scale=(800, 1333)),
)

########################################################################################################################
# Model Settings
########################################################################################################################

model = dict(
    type='RetinaNet',
    pretrained=None,
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50V1',
        weights_path='weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        weight_decay=5e-5
    ),
    neck=dict(
        type='FPN',
        in_channels=[('C2', 256), ('C3', 512), ('C4', 1024), ('C5', 2048)],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        interpolation_method='bilinear',
        weight_decay=5e-5,
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1., 1., 1., 1.],
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        alpha=0.25,
        gamma=2.0,
        label_smoothing=0.05,
        num_pre_nms=1000,
        min_confidence=0.05, 
        nms_threshold=0.75,
        max_instances=100,
        soft_nms_sigma=0.5,
        weight_decay=5e-5
    ),
)
########################################################################################################################
# Training and Test Settings
########################################################################################################################
train_cfg = dict(
    weight_decay=5e-5,
)
test_cfg = dict(
)
