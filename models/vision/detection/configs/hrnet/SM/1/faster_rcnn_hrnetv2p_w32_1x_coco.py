# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import os.path as osp

# date time settings to update paths for jobs
from datetime import datetime
now = datetime.now()
time_str = now.strftime("%d-%m-%Y-%H-%M")
date_str = now.strftime("%d-%m-%Y")


# sagemaker settings
sagemaker_user=dict(
    user_id='mzanur',
    s3_bucket='mzanur-sagemaker',
    docker_image='578276202366.dkr.ecr.us-east-1.amazonaws.com/mzanur-awsdet-ecr:awsdet',
    hvd_processes_per_host=8,
    hvd_instance_type='ml.p3dn.24xlarge', # 'ml.p3.16xlarge',
    hvd_instance_count=4,
)
# settings for distributed training on sagemaker
distributions=dict(
    mpi=dict(
        enabled=True,
        processes_per_host=sagemaker_user['hvd_processes_per_host'],
        custom_mpi_options="-x OMPI_MCA_btl_vader_single_copy_mechanism=none -x TF_CUDNN_USE_AUTOTUNE=0",
    )
)
# sagemaker channels
channels=dict( 
    coco='s3://{}/awsdet/data/coco/'.format(sagemaker_user['s3_bucket']),
    weights='s3://{}/awsdet/data/weights/'.format(sagemaker_user['s3_bucket'])
)

job_str='{}x{}-{}'.format(sagemaker_user['hvd_instance_count'], sagemaker_user['hvd_processes_per_host'], time_str)
sagemaker_job=dict(
    s3_path='s3://{}/faster-rcnn/outputs/{}'.format(sagemaker_user['s3_bucket'], time_str),
    job_name='{}-hrnet-frcnn-{}'.format(sagemaker_user['user_id'], job_str),
    output_path='',
)
sagemaker_job['output_path']='{}/output/{}'.format(sagemaker_job['s3_path'], sagemaker_job['job_name'])


# model settings
model = dict(
    type='FasterRCNN',
    norm_type='BN',
    backbone=dict(
        type='KerasBackbone',
        model_name='HRNetV2p',
        weights_path='hrnet_w32c',
        weight_decay=5e-5
    ),
    neck=dict(
        type='HRFPN',
        in_channels=[('C2', 32), ('C3', 64), ('C4', 128), ('C5', 256)],
        out_channels=256,
        num_outs=5,
        weight_decay=5e-5,
    ),
    rpn_head=dict(
        type='RPNHead',
        anchor_scales=[8.],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds= [1.0, 1.0, 1.0, 1.0],
        feat_channels=512,
        num_samples=256,
        positive_fraction=0.5,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        num_pre_nms_train=6000,
        num_post_nms_train=2000,
        num_pre_nms_test=2000,
        num_post_nms_test=1000,
        weight_decay=5e-5,
        use_smooth_l1=False
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
        min_confidence=0.005,
        nms_threshold=0.75,
        max_instances=100,
        weight_decay=5e-5,
        use_conv=False,
        use_bn=False,
        use_smooth_l1=False,
        soft_nms_sigma=0.5
    ),
)

# model training and testing settings
train_cfg = dict(
    freeze_patterns=['_bn$'],
    weight_decay=5e-5,
    sagemaker=True
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
        preproc_mode='rgb',
        mean=(123.68, 116.78, 103.94),
        std=(1.0, 1.0, 1.0),
        scale=(800, 1344)),
    val=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode='rgb',
        mean=(123.68, 116.78, 103.94),
        std=(1.0, 1.0, 1.0),
        scale=(800, 1344)),
    test=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode='rgb',
        mean=(123.68, 116.78, 103.94),
        std=(1.0, 1.0, 1.0),
        scale=(800, 1344)),
)
# yapf: enable
evaluation = dict(interval=1)
# optimizer
optimizer = dict(
    type='MomentumOptimizer',
    learning_rate=1e-2,
    momentum=0.9,
    nesterov=False,
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
    warmup_iters=500 if sagemaker_user['hvd_instance_count'] == 1 else 1000,
    warmup_ratio=0.001,
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
work_dir = './work_dirs/{}'.format(osp.splitext(osp.basename(__file__))[0])
load_from = None
resume_from = None
workflow = [('train', 1)]
