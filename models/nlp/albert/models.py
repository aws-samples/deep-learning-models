from typing import Optional

import tensorflow as tf
from transformers import (
    PretrainedConfig,
    TFAutoModelForPreTraining,
    TFAutoModelForQuestionAnswering,
)


def get_initializer(stddev):
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)


def load_qa_from_pretrained(
    model: Optional[tf.keras.Model] = None,
    name: Optional[str] = None,
    path: Optional[str] = None,  # path to checkpoint from TF...ForPreTraining
    config: Optional[PretrainedConfig] = None,
) -> tf.keras.Model:
    """ Load a TF...QuestionAnswering model by taking the main layer of a pretrained model. """
    assert (
        bool(name) ^ bool(model) ^ (bool(path) and bool(config))
    ), "Pass either name, model, or (path and config)"

    if name is not None:
        return TFAutoModelForQuestionAnswering.from_pretrained(name)

    elif model is not None:
        pretrained_model = model
    elif path is not None:
        pretrained_model = TFAutoModelForPreTraining.from_config(config)
        pretrained_model.load_weights(path)

    qa_model = TFAutoModelForQuestionAnswering.from_config(pretrained_model.config)
    pretrained_main_layer = getattr(pretrained_model, qa_model.base_model_prefix)
    assert (
        pretrained_main_layer is not None
    ), f"{pretrained_model} has no attribute '{model.base_model_prefix}'"
    # Generalized way of saying `model.albert = pretrained_model.albert`
    setattr(qa_model, qa_model.base_model_prefix, pretrained_main_layer)
    return qa_model
