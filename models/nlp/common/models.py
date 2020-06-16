from typing import Optional

import tensorflow as tf
from transformers import (
    AutoConfig,
    PretrainedConfig,
    TFAutoModelForPreTraining,
    TFAutoModelForQuestionAnswering,
    TFPreTrainedModel,
)

from common.arguments import ModelArguments
from common.utils import create_config


def create_model(model_class, model_args: ModelArguments) -> tf.keras.Model:
    """
    Creates a model with the config specified in model_args.
    If model_args.load_from == "huggingface", it is a pretrained model, and the config must be default.
    """
    # Loading from a checkpoint will need to be done outside this method.
    if model_args.load_from in ["scratch", "checkpoint"]:
        config = create_config(model_args)
        model = model_class.from_config(config)
        return model
    elif model_args.load_from == "huggingface":
        # No custom config if loading a pretrained model
        assert model_args.hidden_dropout_prob == 0 and model_args.pre_layer_norm is None
        model = model_class.from_pretrained(model_args.model_desc)
        return model
    else:
        assert False


def load_qa_from_pretrained(
    model: Optional[tf.keras.Model] = None,
    name: Optional[str] = None,
    path: Optional[str] = None,  # path to checkpoint from TF...ForPreTraining
    config: Optional[PretrainedConfig] = None,
) -> tf.keras.Model:
    """
    Load a TF...QuestionAnswering model by taking the main layer of a pretrained model.
    Preserves the model.config attribute.
    """
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
