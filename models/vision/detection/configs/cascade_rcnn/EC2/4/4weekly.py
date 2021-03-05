base_files = ['../../../common/sagemaker_1x8.py',
              '../../../common/datasets/coco.py',
              '../../../common/lr_policy.py',
              '../../../common/runtime.py',]

# model settings
model = dict(
    type='CascadeRCNN',
    norm_type='BN',
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
        allow_low_quality_matches=True,
        num_pre_nms_train=2000,
        num_post_nms_train=2000,
        num_pre_nms_test=1000,
        num_post_nms_test=1000,
        weight_decay=1e-4,
    ),
    bbox_head=dict(
        type='CascadeHead',
        num_stages=3,
        stage_loss_weights=[0.33, 0.33, 0.33],
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
                weight_decay=1e-4,
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
                weight_decay=1e-4,
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
                weight_decay=1e-4,
                use_conv=False,
                use_bn=False,
                soft_nms_sigma=0.5,
                reg_class_agnostic=True
            )
        ]
    )
)


# optimizer
optimizer = dict(
    _overwrite_=True,
    type='MomentumOptimizer',
    learning_rate=1e-2,
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
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11]
)

total_epochs = 1

work_dir = './work_dirs/cascade_rcnn_r50v1_d_fpn_1x_coco'

