# -*- coding: utf-8 -*-
base_files = ['../../../../common/datasets/coco.py',
              '../../../../common/lr_policy.py',
              '../../../../common/runtime.py',]
# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50V1_d',
        weights_path='/data/weights/resnet50v1_d',
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

total_epochs = 1
work_dir = './work_dirs/mask_rcnn_r50v1_d_fpn_1x_coco'
