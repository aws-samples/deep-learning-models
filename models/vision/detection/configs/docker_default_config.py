# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
########################################################################################################################
# Example configuration file for training with Sagemaker
# Pass to train_sagemaker.py as
# python train_sagemaker.py configs/sagemaker_configuration.py
########################################################################################################################

import os
import pathlib

########################################################################################################################
# Model artifact paths
# Data input and output locations
########################################################################################################################

dataset_type = 'CocoDataset'
data_root = '/workspace/shared_workspace/data/coco/coco'

weights_file = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = '/workspace/shared_workspace/data/weights/{}'.format(weights_file)

os.environ['SM_OUTPUT_DATA_DIR'] = '/workspace/shared_workspace/output'
outputs_path = pathlib.Path(os.getenv('SM_OUTPUT_DATA_DIR')).as_posix()
output_weights = pathlib.Path(os.getenv('SM_OUTPUT_DATA_DIR')).joinpath('faster-rcnn-resnet-50-weights').as_posix()

########################################################################################################################
# Hyperparameters
# More common stuff to adjust
########################################################################################################################

training_epochs = 13
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x'
load_from = None
resume_from = None
loss_weights = {'rpn_class_loss': 1., 
                'rpn_bbox_loss': 1., 
                'rcnn_class_loss': 1., 
                'rcnn_bbox_loss': 1.,
                'reg_loss': 1.}
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
    type='FasterRCNN',
    pretrained=None,
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50V1',
        weights_path=weights_file,
        weight_decay=1e-5,
    ),
    neck=dict(
        type='FPN',
        interpolation_method='bilinear',
        weight_decay=1e-5,
    ),
    rpn_head=dict(
        type='RPNHead',
        anchor_scales=[32, 64, 128, 256, 512],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_feature_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds= [1.0, 1.0, 1.0, 1.0],
        num_rpn_deltas=256,
        positive_fraction=0.5,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        num_pre_nms_train=12000,
        num_post_nms_train=2000,
        num_pre_nms_test=12000,
        num_post_nms_test=2000,
        weight_decay=1e-5,
        padded_img_shape=(1333, 1333),
    ),
    bbox_roi_extractor=dict(
        type='PyramidROIAlign',
        pool_shape=[7, 7],
        pool_type='avg',
        use_tf_crop_and_resize=True,
    ),
    bbox_head=dict(
        type='BBoxHead',
        num_classes=81,
        pool_size=[7, 7],
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        min_confidence=0.001,
        nms_threshold=0.7,
        max_instances=100,
        weight_decay=1e-5,
        use_conv=True,
        label_smoothing=0.0,
        use_bn=False,
        soft_nms_sigma=0.5, # 0.0 = hard nms
    )
)

########################################################################################################################
# Training and Test Settings
########################################################################################################################

train_cfg = dict(
    weight_decay=1e-5,    
)

test_cfg = dict(
)
