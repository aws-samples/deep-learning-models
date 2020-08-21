# -*- coding: utf-8 -*-
base_files = ['../../../common/datasets/coco.py',
              '../../../common/lr_policy.py',
              '../../../common/runtime.py',
              '../../../common/models/mask_rcnn_fpn.py']
# overwrite dataset mean and std
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/COCO/'
preproc_mode = 'caffe'
image_mean = (123.68, 116.78, 103.94)
image_std = (1., 1., 1.)
data = dict(
    _overwrite_ = True,
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

# overwrite default optimizer
optimizer = dict(
    _overwrite_=True,
    type='SGD',
    learning_rate=1e-2,
    momentum=0.9,
    nesterov=False,
)

# learning policy
lr_config = dict(
    _overwrite_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1, outdir='checkpoints')

work_dir = './work_dirs/mask_rcnn_r50_fpn_1x_coco'

