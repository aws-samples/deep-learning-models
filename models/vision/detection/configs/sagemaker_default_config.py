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
from mmdet.utils.schedulers.schedulers import WarmupScheduler
hvd.init()

#######################
# Temp Used for testing locally
#######################
'''os.environ['SM_OUTPUT_DATA_DIR'] = '/tmp/output'
data_root = '/workspace/shared_workspace/data/coco'
weights_file = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = '/workspace/shared_workspace/autograph/mmdetection_tf/configs/weights/{}'.format(weights_file)
'''
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

batch_size_per_device = 4
workers_per_gpu = 2
global_batch_size = batch_size_per_device*hvd.size()
training_epochs = 12
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
# learning rate of .01 for batch size of 8 is standard
# with linear scaling for batch size adjustments
base_learning_rate = 5e-3
scaled_learning_rate = base_learning_rate*global_batch_size/8
coco_images = 117504
steps_per_epoch = coco_images//global_batch_size
# scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay([steps_per_epoch*8], 
#                                                                  [scaled_learning_rate, scaled_learning_rate/10])
# apply learning rate warmup
# initial_warmup_rate = scaled_learning_rate/4
# warmup_steps = steps_per_epoch//4
# scheduler = WarmupScheduler(scheduler, initial_warmup_rate, warmup_steps)

# scheduler = tfa.optimizers.TriangularCyclicalLearningRate(base_learning_rate/8, scaled_learning_rate, 2000)
scheduler = tf.keras.experimental.CosineDecayRestarts(scaled_learning_rate, 1000)
scheduler = WarmupScheduler(scheduler, base_learning_rate, 1000)

optimizer = tf.keras.optimizers.SGD(scheduler, momentum=0.9, nesterov=False)

fp16 = True
if fp16:
    # optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale="dynamic")
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scale='dynamic')

########################################################################################################################
# Checkpoints and logging
########################################################################################################################
checkpoint_interval = 1
evaluation_interval = 1
########################################################################################################################
# Dataset Settings
########################################################################################################################

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1213, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1216, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
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
        std=(1., 1., 1.), # (58.395, 57.12, 57.375), 
        scale=(800, 1216)),
    val=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode='caffe',
        mean=(123.675, 116.28, 103.53),
        std=(1., 1., 1.), # (58.395, 57.12, 57.375), 
        scale=(800, 1216)),
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
        scale=(800, 1216)),
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
        weight_decay=1e-5
    ),
    neck=dict(
        type='FPN',
        weight_decay=1e-5,
        ),
    rpn_head=dict(
        type='RPNHead',
        anchor_scales=[32, 64, 128, 256, 512],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_feature_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds= [1.0, 1.0, 2.0, 2.0], #[1.0, 1.0, 1.0, 1.0],
        num_rpn_deltas=256,
        positive_fraction=0.5,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        num_pre_nms_train=12000,
        num_post_nms_train=2000,
        num_pre_nms_test=12000,
        num_post_nms_test=2000,
        weight_decay=1e-5,
        padded_img_shape=(1216, 1216)
    ),
    bbox_roi_extractor=dict(
    type='PyramidROIAlign',
    pool_shape=[7, 7]),
    bbox_head=dict(
    type='BBoxHead',
    num_classes=81,
    pool_size=[7, 7],
    target_means=[0., 0., 0., 0.],
    target_stds=[0.1, 0.1, 0.2, 0.2],
    min_confidence=0.001,
    nms_threshold=0.5,
    max_instances=100,
    weight_decay=1e-5,
    use_conv=True)
)

########################################################################################################################
# Training and Test Settings
########################################################################################################################

train_cfg = dict(
    freeze_patterns=('^conv[12]',), # '_bn$'), # freeze upto stage 1 (conv2 block)
    weight_decay=1e-5,
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
