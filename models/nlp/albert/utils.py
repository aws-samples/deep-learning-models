import logging
from functools import lru_cache
from typing import List

import tensorflow as tf
import tqdm
from transformers import AlbertTokenizer, PreTrainedTokenizer
from transformers.data.processors.squad import (
    SquadExample,
    SquadFeatures,
    SquadProcessor,
    SquadResult,
    SquadV2Processor,
    squad_convert_examples_to_features,
)

# See https://github.com/huggingface/transformers/issues/3782; this import must come last
import horovod.tensorflow as hvd  # isort:skip


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
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int64) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    output_tensor = tf.reshape(output_tensor, [batch_size, -1, width])
    return output_tensor


def gather_indexes_2d(sequence_tensor: "[batch,seq_length]", positions) -> tf.Tensor:
    """ Gathers the vectors at the specific positions over a 2D minibatch."""
    # TODO: Merge this with gather_indexes()
    sequence_shape = sequence_tensor.shape.as_list()
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int64) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    output_tensor = tf.reshape(output_tensor, [batch_size, -1])
    return output_tensor


@lru_cache(maxsize=4)
def get_dataset(
    tokenizer: PreTrainedTokenizer,
    processor: SquadProcessor,
    data_dir: str,
    filename: str,
    batch_size: int,
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
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        return dataset


def get_tokenizer():
    return AlbertTokenizer.from_pretrained("albert-base-v2")
