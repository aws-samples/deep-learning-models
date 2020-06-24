# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

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

sagemaker_job=dict(
    s3_path='s3://{}/faster-rcnn/outputs/{}'.format(sagemaker_user['s3_bucket'], time_str),
    job_name='{}-frcnn-{}'.format(sagemaker_user['user_id'], time_str),
    output_path='',
)
sagemaker_job['output_path']='{}/output/{}'.format(sagemaker_job['s3_path'], sagemaker_job['job_name'])

# model settings
model=dict(
    type='FasterRCNN',
    pretrained=None,
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50V1',
        weights_path='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', # will be fully resolved to path in train script
        weight_decay=1e-5
    ),
    neck=dict(
        type='FPN',
        in_channels=[('C2', 256), ('C3', 512), ('C4', 1024), ('C5', 2048)],
        out_channels=256,
        num_outs=5,
        interpolation_method='bilinear',
        weight_decay=1e-5,
    ),
    rpn_head=dict(
        type='RPNHead',
        anchor_scales=[8.],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        feat_channels=512,
        num_samples=256,
        positive_fraction=0.5,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        num_pre_nms_train=12000,
        num_post_nms_train=2000,
        num_pre_nms_test=12000,
        num_post_nms_test=2000,
        weight_decay=1e-5,
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
    nms_threshold=0.5,
    max_instances=100,
    weight_decay=1e-5,
    use_conv=True,
    use_bn=False,
    soft_nms_sigma=0.5)
)

# model training and testing settings
train_cfg=dict(
    weight_decay=1e-5,
    sagemaker=True
)
test_cfg=dict(
)

# dataset settings
dataset_type='CocoDataset'
data_root='/data/COCO/' # will be resolved to SM specific path in train.py
data=dict(
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
evaluation=dict(interval=1)
# optimizer
optimizer=dict(
    type='SGD',
    learning_rate=1e-2,
    momentum=0.9,
    nesterov=False,
)
# extra options related to optimizers
optimizer_config=dict(
    amp_enabled=True,
)
# learning policy
lr_config=dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
#TODO: add support for S3 checkpointing
checkpoint_config=dict(
    interval=1,
    outdir='checkpoints')
# log, tensorboard configuration
log_config=dict(
    interval=50,
    hooks=[
        dict(
            type='TextLoggerHook'
        ),
        dict(
            type='TensorboardLoggerHook',
            log_dir=None,
            image_interval=100,
            s3_dir='{}/tensorboard/{}'.format(sagemaker_job['s3_path'], sagemaker_job['job_name'])
        ),
        dict(
            type='Visualizer',
            dataset_cfg=data['val'],
            interval=100,
            top_k=10,
            run_on_sagemaker=True,
        ),
    ]
)
# runtime settings
total_epochs=12
log_level='INFO'
work_dir='./work_dirs/faster_rcnn_r50_fpn_1x_coco'
load_from=None
resume_from=None
workflow=[('train', 1)]
