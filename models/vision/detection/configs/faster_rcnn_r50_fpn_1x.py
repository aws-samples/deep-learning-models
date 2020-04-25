# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# model settings
model = dict(
    type='FasterRCNN',
    pretrained=None,
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50',
        weights_path='weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    ),
    neck=dict(
        type='FPN',
        #in_channels=[256, 512, 1024, 2048],
        #out_channels=256,
        #num_outs=5
        ),
    rpn_head=dict(
        type='RPNHead',
        # in_channels=256,
        # feat_channels=256,
        anchor_scales=[32, 64, 128, 256, 512],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_feature_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds= [1.0, 1.0, 1.0, 1.0],#  [0.1, 0.1, 0.2, 0.2],
        num_rpn_deltas=256,
        positive_fraction=0.5,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        num_pre_nms_train=48000,
        num_post_nms_train=2000,
        num_pre_nms_test=12000,
        num_post_nms_test=2000,
        # loss_cls=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
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
    min_confidence=0.005, #0.7,
    nms_threshold=0.3,
    max_instances=100)
)
# model training and testing settings
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
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/COCO/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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
    imgs_per_gpu=1,
    workers_per_gpu=2, # TODO: unused
    train=dict(
        type=dataset_type,
        train=True,
        dataset_dir=data_root,
        subset='train',
        flip_ratio=0.5,
        pad_mode='fixed',
        mean=(123.675, 116.28, 103.53),
        std=(1., 1., 1.), # (58.395, 57.12, 57.375), 
        scale=(800, 1333)),
    val=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        mean=(123.675, 116.28, 103.53),
        std=(1., 1., 1.), # (58.395, 57.12, 57.375), 
        scale=(800, 1333)),
    test=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        mean=(123.675, 116.28, 103.53),
        std=(1., 1., 1.),
        scale=(800, 1333)),
)
# yapf: enable
evaluation = dict(interval=1)
# optimizer
optimizer = dict(
    type='SGD',
#    weight_decay=0, #1e-4,
    learning_rate=5e-3,
    momentum=0.9,
    nesterov=False,
    clipnorm=10, # not supported for LossScaleOptimizer wrapper(why?)
    #clipvalue=2,
)
# extra options related to optimizers
optimizer_config = dict(
    amp_enabled=False,
)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1, outdir='checkpoints')
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir='/tmp/tensorboard')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl') #unused
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
