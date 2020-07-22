from typing import List

import tensorflow as tf


def get_dataset_from_tfrecords(
    *,
    model_type: str,
    filenames: List[str],
    per_gpu_batch_size: int,
    max_seq_length: int,
    max_predictions_per_seq: int = None,
    buffer_size: int = 1000,
    shard: bool = True,
) -> "tf.data.Dataset":
    """ Reads the dataset from TFRecords and returns it.
    Returns a dataset that includes batching, but not gradient accumulation.
    """

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, name_to_features)

    if model_type in ["albert", "bert"]:
        assert max_predictions_per_seq is not None, "Pass --max_predictions_per_seq"
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature(
                [max_seq_length], tf.int64
            ),  # corresponds to input_ids
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
    elif model_type in ["electra"]:
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature(
                [max_seq_length], tf.int64
            ),  # corresponds to input_ids
            "token_type_ids": tf.io.FixedLenFeature(
                [max_seq_length], tf.int64
            ),  # corresponds to token_type_ids
            "attention_mask": tf.io.FixedLenFeature(
                [max_seq_length], tf.int64
            ),  # corresponds to attention_mask
        }
    else:
        raise ValueError(f"model_type={model_type} must be one of ['albert', 'bert', 'electra']")

    # Example input pipeline here: https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/run_pretraining.py#L443
    assert len(filenames) > 0, f"Filenames is an empty list"
    # Shard and shuffle the filenames
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shard:
        import horovod.tensorflow as hvd

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
    dataset = dataset.batch(per_gpu_batch_size, drop_remainder=True)
    # Shuffle the batches and prefetch some batches
    dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    return dataset
