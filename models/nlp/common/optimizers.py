import tensorflow_addons as tfa

from common.learning_rate_schedules import LinearWarmupPolyDecaySchedule


def get_lr_schedule(train_args):
    return LinearWarmupPolyDecaySchedule(
        max_learning_rate=train_args.learning_rate,
        end_learning_rate=train_args.end_learning_rate,
        warmup_steps=train_args.warmup_steps,
        total_steps=train_args.total_steps,
        power=train_args.learning_rate_decay_power,
    )


def get_lamb_optimizer(train_args):
    lr_schedule = get_lr_schedule(train_args)
    wd_schedule = train_args.weight_decay  # TODO: Get weight decay schedule working.
    # LAMB made available in v0.7.0 of tfa, which only works with TF 2.1+.
    optimizer = tfa.optimizers.LAMB(
        learning_rate=lr_schedule,
        weight_decay_rate=wd_schedule,
        beta_1=train_args.beta_1,
        beta_2=train_args.beta_2,
        epsilon=train_args.epsilon,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    )
    return optimizer


def get_adamw_optimizer(train_args):
    # Track issue status in https://github.com/tensorflow/addons/issues/1903
    raise ValueError(
        "This does not work currently, due to the lack of an `exclude_from_weight_decay`"
        " argument in the tfa.optimizers.AdamW implementation. It also does not work without"
        "an explicit weight decay schedule that matches the learning rate schedule."
    )
    # TODO: Get weight decay schedule working. See https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW
    # This causes the model to diverge if learning rate is too low.
    # It appears that ALBERT has a fixed weight decay: https://github.com/google-research/albert/blob/master/optimization.py#L78
    lr_schedule = get_lr_schedule(train_args)
    wd_schedule = train_args.weight_decay  # TODO: Get weight decay schedule working.
    optimizer = tfa.optimizers.AdamW(
        weight_decay=wd_schedule,
        learning_rate=lr_schedule,
        beta_1=train_args.beta_1,
        beta_2=train_args.beta_2,
        epsilon=train_args.epsilon,
        # exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"], # TODO: Implement this
    )
    return optimizer
