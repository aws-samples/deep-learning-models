base_files = ['../../../common/sagemaker_1x8.py',
              '../../../common/datasets/coco.py',
              '../../../common/lr_policy.py',
              '../../../common/runtime.py',
              '../../../common/models/faster_rcnn_fpn.py']

# overwrite train cfg to indicate sagemaker training
train_cfg = dict(
    _overwrite_=True,
    freeze_patterns=['^conv[12]_*', '_bn$'],
    weight_decay=1e-5,
    sagemaker=True,
)

# overwrite default optimizer
optimizer = dict(
    _overwrite_=True,
    type='SGD',
    learning_rate=1e-2,
    momentum=0.9,
    nesterov=False,
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

work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_coco'
