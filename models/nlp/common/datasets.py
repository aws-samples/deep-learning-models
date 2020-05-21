from typing import List

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf


def get_synthetic_mlm_dataset(batch_size: int) -> tf.data.Dataset:
    """ Returns a dataset that includes batching, but not gradient accumulation. """

    def gen(batch_size):
        seq_shape = [batch_size, 512]
        preds_shape = [batch_size, 20]
        input_ids = tf.constant(np.random.randint(10, size=seq_shape), dtype=tf.int64)
        attention_mask = tf.constant(np.random.randint(2, size=seq_shape), dtype=tf.int64)
        token_type_ids = tf.constant(np.random.randint(2, size=seq_shape), dtype=tf.int64)
        masked_lm_positions = tf.constant(np.random.randint(10, size=preds_shape), dtype=tf.int64)
        masked_lm_ids = tf.constant(np.random.randint(2, size=preds_shape), dtype=tf.int64)
        masked_lm_weights = tf.constant(np.random.randint(2, size=preds_shape), dtype=tf.float32)
        next_sentence_labels = tf.constant(np.random.randint(2, size=[batch_size]), dtype=tf.int64)

        input_dict = {
            "input_ids": input_ids,
            "input_mask": attention_mask,
            "segment_ids": token_type_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_weights": masked_lm_weights,
            "next_sentence_labels": next_sentence_labels,
        }

        yield input_dict

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types={
            "input_ids": tf.int64,
            "input_mask": tf.int64,
            "segment_ids": tf.int64,
            "masked_lm_positions": tf.int64,
            "masked_lm_ids": tf.int64,
            "masked_lm_weights": tf.float32,
            "next_sentence_labels": tf.int64,
        },
        args=(batch_size,),
    )
    dataset = dataset.repeat()

    return dataset


def get_mlm_dataset(
    *,
    filenames: List[str],
    max_seq_length: int,
    max_predictions_per_seq: int,
    batch_size: int,
    buffer_size: int = 1000,
) -> tf.data.Dataset:
    """ Reads the dataset from TFRecords and returns it.
    Returns a dataset that includes batching, but not gradient accumulation.
    """

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, name_to_features)

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),  # corresponds to input_ids
        "input_mask": tf.io.FixedLenFeature(
            [max_seq_length], tf.int64
        ),  # corresponds to attention_mask
        "segment_ids": tf.io.FixedLenFeature(
            [max_seq_length], tf.int64
        ),  # corresponds to token_type_ids
        "masked_lm_positions": tf.io.FixedLenFeature(
            [max_predictions_per_seq], tf.int64
        ),  # The number in the sequence that is masked, in range [0, max_seq_length]. 0 signifies a pad.
        "masked_lm_ids": tf.io.FixedLenFeature(
            [max_predictions_per_seq], tf.int64
        ),  # The token id that is masked, in range [0, vocab_size]. 0 signifies a pad.
        "masked_lm_weights": tf.io.FixedLenFeature(
            [max_predictions_per_seq], tf.float32
        ),  # 1 if useful, 0 signifies a pad token
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }

    # Example input pipeline here: https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/run_pretraining.py#L443
    # 2048 TFRecord files here
    assert len(filenames) > 0, f"Filenames is an empty list"
    # Shard and shuffle the filenames
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shard(hvd.size(), hvd.rank())
    dataset = dataset.shuffle(buffer_size=len(filenames), reshuffle_each_iteration=True)
    dataset = dataset.repeat()

    # `cycle_length` is the number of parallel files that get read
    num_cpu_threads = 2 * 96
    cycle_length = min(num_cpu_threads, len(filenames))
    # file_to_dataset_func = lambda file: tf.data.TFRecordDataset(file).map(_parse_function)
    file_to_dataset_func = lambda file: tf.data.TFRecordDataset(file)
    dataset = dataset.interleave(
        file_to_dataset_func,
        cycle_length=cycle_length,
        block_length=1,
        num_parallel_calls=cycle_length,
    )
    # Map and batch will be automatically fused together, see https://www.tensorflow.org/api_docs/python/tf/data/experimental/map_and_batch
    dataset = dataset.map(_parse_function, num_parallel_calls=num_cpu_threads)
    dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # Shuffle the batches and prefetch some batches
    dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    return dataset
