# model training and testing settings
train_cfg = dict(
    freeze_patterns=['^conv[12]_*', '_bn$'],
    weight_decay=1e-4,
)
test_cfg = dict(
)

# run eval on validation with interval specified below
evaluation = dict(interval=1)


checkpoint_config = dict(interval=1, outdir='checkpoints')

# logging and viz options
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir='/tmp/tensorboard')
    ])

# runtime settings
total_epochs = 12
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

