import tensorflow_addons as tfa
import transformers

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
    """
    When using AdamW with transformers, we must exclude LayerNorm and bias from weight decay.
    Not doing so keeps MLM accuracy under 0.15. TensorFlow only has Adam, TensorFlow Addons has AdamW
    but does not include an exclude_from_weight_decay option. The BERT and ALBERT google-research
    implementations have an AdamWeightDecay implementation, but it is for TF1. The transformers
    implementation is the only TF2 one with exclude_from_weight_decay I have found.

    TODO: If weight decay is decoupled from the gradient update, then we need to decay the weight decay
    rate along the same schedule as the learning rate. No serious implementations seem to do this yet.
    Would be a good PR for huggingface.

    TODO: Unclear if LAMB needs a weight decay schedule. If so, that would explain why performance gets
    worse when the learning rate is too low.

    TF-Addons issue: https://github.com/tensorflow/addons/issues/1903
    AdamW note on weight decay *schedule*: https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW
    ALBERT does not use a weight decay schedule: https://github.com/google-research/albert/blob/master/optimization.py#L78
    """
    lr_schedule = get_lr_schedule(train_args)
    wd_schedule = train_args.weight_decay

    optimizer = transformers.AdamWeightDecay(
        learning_rate=lr_schedule,
        beta_1=train_args.beta_1,
        beta_2=train_args.beta_2,
        epsilon=train_args.epsilon,
        weight_decay_rate=wd_schedule,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    )
    return optimizer
