# optimizer
optimizer = dict(
    type='MomentumOptimizer',
    learning_rate=1e-2,
    momentum=0.9,
    nesterov=False,
)

# extra options related to optimizers
optimizer_config = dict(
    amp_enabled=True,
)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]
)

