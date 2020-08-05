import logging
import os
from functools import lru_cache
from typing import List

import tensorflow as tf
import tqdm
from transformers import (
    AlbertTokenizer,
    AutoConfig,
    BertTokenizer,
    BertTokenizerFast,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.data.processors.squad import (
    SquadExample,
    SquadFeatures,
    SquadProcessor,
    SquadResult,
    SquadV2Processor,
    squad_convert_examples_to_features,
)

from common.arguments import ModelArguments

# See https://github.com/huggingface/transformers/issues/3782; this import must come last
import horovod.tensorflow as hvd  # isort:skip

try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn(
            "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable."
        )
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def rewrap_tf_function(func, experimental_compile=None):
    # If func is already a tf.function, un-wrap it and re-wrap it.
    # Necessary to avoid TF's global cache bugs when changing models.
    if hasattr(func, "python_function"):
        # printing func._list_all_concrete_functions_for_serialization() here is always an empty list
        return tf.function(func.python_function, experimental_compile=experimental_compile)
    else:
        return tf.function(func, experimental_compile=experimental_compile)


def f1_score(precision: tf.Tensor, recall: tf.Tensor) -> tf.Tensor:
    return tf.math.maximum(
        tf.cast(0, dtype=precision.dtype), 2 * (precision * recall) / (precision + recall)
    )


def gather_indexes(sequence_tensor: "[batch,seq_length,width]", positions) -> tf.Tensor:
    """Gathers the vectors at the specific positions over a 3D minibatch."""
    sequence_shape = sequence_tensor.shape.as_list()
    per_gpu_batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(tf.range(0, per_gpu_batch_size, dtype=tf.int64) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [per_gpu_batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    output_tensor = tf.reshape(output_tensor, [per_gpu_batch_size, -1, width])
    return output_tensor


def gather_indexes_2d(sequence_tensor: "[batch,seq_length]", positions) -> tf.Tensor:
    """ Gathers the vectors at the specific positions over a 2D minibatch."""
    # TODO: Merge this with gather_indexes()
    sequence_shape = sequence_tensor.shape.as_list()
    per_gpu_batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]

    flat_offsets = tf.reshape(tf.range(0, per_gpu_batch_size, dtype=tf.int64) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [per_gpu_batch_size * seq_length])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    output_tensor = tf.reshape(output_tensor, [per_gpu_batch_size, -1])
    return output_tensor


@lru_cache(maxsize=4)
def get_dataset(
    tokenizer: PreTrainedTokenizer,
    processor: SquadProcessor,
    data_dir: str,
    filename: str,
    per_gpu_batch_size: int,
    shard: bool,
    drop_remainder: bool,
    shuffle: bool = True,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    max_query_length: int = 64,
    evaluate: bool = False,
    return_raw_features: bool = False,
    repeat: bool = False,
) -> tf.data.Dataset:
    # Convert the data from a JSON file into a tf.data.Dataset
    # This function should also work to fetch the val_dataset
    if evaluate:
        examples: List[SquadExample] = processor.get_dev_examples(data_dir, filename=filename)
    else:
        examples: List[SquadExample] = processor.get_train_examples(data_dir, filename=filename)
    # dataset is a tuple of (features, dataset)
    dataset: List[SquadFeatures] = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=not evaluate,
        return_dataset=None if return_raw_features else "tf",
        threads=16,
    )
    if return_raw_features:
        return dataset
    else:
        if shard:
            dataset = dataset.shard(hvd.size(), hvd.rank())
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(per_gpu_batch_size, drop_remainder=drop_remainder)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        return dataset


def create_config(model_args: ModelArguments) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(model_args.model_desc)
    config.pre_layer_norm = model_args.pre_layer_norm
    config.hidden_dropout_prob = model_args.hidden_dropout_prob
    config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob
    return config


def create_tokenizer(model_type: str) -> PreTrainedTokenizer:
    if model_type == "albert":
        return AlbertTokenizer.from_pretrained("albert-base-v2")
    elif model_type == "bert":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_type == "electra":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        raise ValueError(f"model_type={model_type} must be one of ['albert', 'bert', 'electra']")
