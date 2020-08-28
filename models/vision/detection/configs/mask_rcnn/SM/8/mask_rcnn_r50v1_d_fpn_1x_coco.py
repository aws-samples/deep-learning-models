# -*- coding: utf-8 -*-
base_files = ['../../../common/datasets/coco.py',
              '../../../common/lr_policy.py',
              '../../../common/runtime.py',]
# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50V1_d',
        weights_path='resnet50v1_d',
        weight_decay=1e-4
    ),
    neck=dict(
        type='FPN',
        in_channels=[('C2', 256), ('C3', 512), ('C4', 1024), ('C5', 2048)],
        out_channels=256,
        num_outs=5,
        interpolation_method='bilinear',
        weight_decay=1e-4,
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
        weight_decay=1e-4,
    ),
    bbox_roi_extractor=dict(
        type='PyramidROIAlign',
        pool_shape=[7, 7],
        pool_type='avg',
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
        weight_decay=1e-4,
        use_conv=False,
        use_bn=False,
        soft_nms_sigma=0.5
    ),
    mask_head=dict(
        type='MaskHead',
        num_classes=81,
        weight_decay=1e-5,
        use_bn=False,
    ),
    mask_roi_extractor=dict(
        type='PyramidROIAlign',
        pool_shape=[14, 14],
        pool_type='avg',
    ),
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/COCO/'
preproc_mode = 'rgb'
image_mean = (123.68, 116.78, 103.94)
image_std = (58.393, 57.12, 57.375)

data = dict(
    _overwrite_=True,
    imgs_per_gpu=4,
    train=dict(
        type=dataset_type,
        train=True,
        dataset_dir=data_root,
        subset='train',
        flip_ratio=0.5,
        pad_mode='fixed',
        preproc_mode=preproc_mode,
        mean=image_mean,
        std=image_std,
        scale=(800, 1333),
        mask=True),
    val=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode=preproc_mode,
        mean=image_mean,
        std=image_std,
        scale=(800, 1333),
        mask=True),
    test=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode=preproc_mode,
        mean=image_mean,
        std=image_std,
        scale=(800, 1333),
        mask=True),
)

# overwrite train cfg to indicate sagemaker training
train_cfg = dict(
    _overwrite_=True,
    freeze_patterns=['^conv[12]_*', '_bn$'],
    weight_decay=1e-4,
    sagemaker=True,
)

# optimizer
optimizer = dict(
    _overwrite_=True,
    type='MomentumOptimizer',
    learning_rate=5e-3,
    momentum=0.9,
    nesterov=False,
)

# extra options related to optimizers
optimizer_config = dict(
    _overwrite_=True,
    amp_enabled=True,
    gradient_clip=5.0,
)

# overwrite lr policy
# learning policy
lr_config = dict(
    _overwrite_=True,
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[8, 11]
)

# log, tensorboard configuration with s3 path for logs
log_config=dict(
    _overwrite_=True,
    interval=50,
    hooks=[
        dict(
            type='TextLoggerHook'
        ),
        dict(
            type='TensorboardLoggerHook',
            log_dir=None,
            image_interval=100,
            s3_dir='', # set dynamically
        ),
        dict(
            type='Visualizer',
            dataset_cfg=None, # set dynamically
            interval=100,
            top_k=10,
            run_on_sagemaker=True,
        ),
    ]
)

work_dir = './work_dirs/mask_rcnn_r50v1_d_fpn_1x_coco'
