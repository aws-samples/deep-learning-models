base_files = ['../../../common/sagemaker_4x8.py',
              '../../../common/datasets/coco.py',
              '../../../common/lr_policy.py',
              '../../../common/runtime.py',
              '../../../common/models/faster_rcnn_fpn.py']

# overwrite default optimizer
optimizer = dict(
    _overwrite_=True,
    type='SGD',
    learning_rate=1e-2,
    momentum=0.9,
    nesterov=False,
)

# log, tensorboard configuration with S3 path for logs
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


work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_coco'
