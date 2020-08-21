# -*- coding: utf-8 -*-
base_files = ['../../../common/datasets/coco.py',
              '../../../common/lr_policy.py',
              '../../../common/runtime.py',]

model = dict(
    type='RetinaNet',
    pretrained=None,
    norm_type='BN',
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50V1_b',
        weights_path='weights/resnet50v1_b', # SavedModel format
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
        allow_low_quality_matches=True,
        alpha=0.25,
        gamma=2.0,
        label_smoothing=0.0,
        num_pre_nms=1000,
        min_confidence=0.05,
        nms_threshold=0.75, # using soft nms
        max_instances=100,
        soft_nms_sigma=0.5,
        weight_decay=1e-4
    ),
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

# runtime settings
work_dir = './work_dirs/retinanet_r50v1_b_fpn_1x_coco'
