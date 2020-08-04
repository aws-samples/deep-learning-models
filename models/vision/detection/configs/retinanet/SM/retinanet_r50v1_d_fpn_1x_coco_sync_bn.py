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
    hvd_instance_type='ml.p3.16xlarge',
    hvd_instance_count=1,
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
    s3_path='s3://{}/retinanet/outputs/{}'.format(sagemaker_user['s3_bucket'], time_str),
    job_name='{}-retinanet-{}'.format(sagemaker_user['user_id'], job_str),
    output_path='',
)
sagemaker_job['output_path']='{}/output/{}'.format(sagemaker_job['s3_path'], sagemaker_job['job_name'])



# model settings
model = dict(
    type='RetinaNet',
    pretrained=None,
    norm_type='SyncBN',
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50V1_d',
        weights_path='resnet50v1_d', # SavedModel format
        weight_decay=1e-4
    ),
    neck=dict(
        type='FPN',
        in_channels=[('C2', 256), ('C3', 512), ('C4', 1024), ('C5', 2048)],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        interpolation_method='bilinear',
        weight_decay=1e-4,
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
        label_smoothing=0.0,
        num_pre_nms=1000,
        min_confidence=0.005,
        nms_threshold=0.75, # using soft nms
        max_instances=100,
        soft_nms_sigma=0.5,
        weight_decay=1e-4
    ),
)
# model training and testing settings
train_cfg = dict(
    freeze_patterns=['^conv[12]_*'],
    weight_decay=1e-4,
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
        std=(58.393, 57.12, 57.375),
        scale=(800, 1333)),
    val=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode='rgb',
        mean=(123.68, 116.78, 103.94),
        std=(58.393, 57.12, 57.375),
        scale=(800, 1333)),
    test=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode='rgb',
        mean=(123.68, 116.78, 103.94),
        std=(58.393, 57.12, 57.375),
        scale=(800, 1333)),
)
# yapf: enable
evaluation = dict(interval=1)
# optimizer
optimizer = dict(
    type='MomentumOptimizer',
    learning_rate=5e-3,
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
    warmup_iters=500, 
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

