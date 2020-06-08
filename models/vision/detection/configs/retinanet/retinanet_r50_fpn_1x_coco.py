# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

# model settings
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
        interpolation_method='nearest',
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
        nms_threshold=0.75, # using soft nms
        max_instances=100,
        soft_nms_sigma=0.5,
        weight_decay=5e-5
    ),
)
# model training and testing settings
train_cfg = dict(
    weight_decay=5e-5,
)
test_cfg = dict(
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/COCO/'
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
# yapf: enable
evaluation = dict(interval=1)
# optimizer
optimizer = dict(
    type='SGD',
    learning_rate=5e-3,
    momentum=0.9,
    nesterov=True, #False,
)
# extra options related to optimizers
optimizer_config = dict(
    amp_enabled=True,
    gradient_clip=10.0,
)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500, 
    warmup_ratio=1.0 / 10,
    step=[8, 11])
checkpoint_config = dict(interval=1, outdir='checkpoints')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir='/tmp/tensorboard')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
log_level = 'INFO'
work_dir = './work_dirs/retinanet_r50_fpn_1x_amp_bn'
load_from = None
resume_from = None
workflow = [('train', 1)]

