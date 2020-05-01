from functools import lru_cache
from typing import List

import tensorflow as tf
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


def rewrap_tf_function(func):
    # If func is already a tf.function, un-wrap it and re-wrap it.
    # Necessary to avoid TF's global cache bugs when changing models.
    if hasattr(func, "python_function"):
        return tf.function(func.python_function)
    else:
        return tf.function(func)


class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.
    From tensorflow_datasets
    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


def f1_score(precision: tf.Tensor, recall: tf.Tensor) -> tf.Tensor:
    return tf.math.maximum(
        tf.cast(0, dtype=precision.dtype), 2 * (precision * recall) / (precision + recall)
    )


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
