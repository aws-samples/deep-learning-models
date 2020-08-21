# -*- coding: utf-8 -*-
# model settings
model = dict(
    type='CascadeRCNN',
    norm_type='BN',
    backbone=dict(
        type='KerasBackbone',
        model_name='ResNet50V1',
        weights_path='weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
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
        target_stds= [1.0, 1.0, 1.0, 1.0],
        feat_channels=512,
        num_samples=256,
        positive_fraction=0.5,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        allow_low_quality_matches=True,
        num_pre_nms_train=2000,
        num_post_nms_train=2000,
        num_pre_nms_test=1000,
        num_post_nms_test=1000,
        weight_decay=1e-5,
    ),
    bbox_head=dict(
        type='CascadeHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        iou_thresholds=[0.5, 0.6, 0.7],
        reg_class_agnostic=True,
        bbox_roi_extractor=dict(
            type='PyramidROIAlign',
            pool_shape=[7, 7],
            pool_type='avg'
        ),
        bbox_head=[
            dict(
                type='BBoxHead',
                num_classes=81,
                pool_size=[7, 7],
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2],
                min_confidence=0.005, 
                nms_threshold=0.75,
                weight_decay=1e-5,
                use_conv=False,
                use_bn=False,
                soft_nms_sigma=0.5,
                reg_class_agnostic=True
            ),
            dict(
                type='BBoxHead',
                num_classes=81,
                pool_size=[7, 7],
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1],
                min_confidence=0.005, 
                nms_threshold=0.75,
                weight_decay=1e-5,
                use_conv=False,
                use_bn=False,
                soft_nms_sigma=0.5,
                reg_class_agnostic=True
            ),
            dict(
                type='BBoxHead',
                num_classes=81,
                pool_size=[7, 7],
                target_means=[0., 0., 0., 0.],
                target_stds=[0.033, 0.033, 0.067, 0.067],
                min_confidence=0.005, 
                nms_threshold=0.75,
                weight_decay=1e-5,
                use_conv=False,
                use_bn=False,
                soft_nms_sigma=0.5,
                reg_class_agnostic=True
            )
        ]
    )
)

