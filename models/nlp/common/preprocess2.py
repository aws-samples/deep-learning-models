def remove_none_values(example):
    return example["text"] != ""


def tokenize(example):
    return tokenizer(
        example["text"], padding=True, truncation=True, max_length=data_args.max_seq_length
    )


def get_nlp_dataset(name: str, split: str, shard: bool = True, from_cache: bool = True):
    text_cache_file_name = f"{CACHE_DIR}/{name}-{split}-text.cache"
    tokens_cache_file_name = f"{CACHE_DIR}/{name}-{split}-tokens.cache"
    if not from_cache:
        dset = nlp.load_dataset("wikitext", f"{name}-raw-v1", split=split, cache_dir=CACHE_DIR)
        # We cache the raw text dataset
        dset = dset.filter(remove_none_values, cache_file_name=text_cache_file_name)
        # Now we cache the tokenized dataset
        dset = dset.map(
            tokenize, batched=True, batch_size=1000, cache_file_name=tokens_cache_file_name
        )

    dset = nlp.Dataset.from_file(tokens_cache_file_name)

    # Then shard
    if shard:
        cache_file_name = (
            # f"/root/{name}-{split}-size{hvd.size()}-rank{hvd.rank()}.cache"
            f"{CACHE_DIR}/shards/{name}-{split}-size{hvd.size()}-rank{hvd.rank()}.cache"
        )
        dset = dset.shard(hvd.size(), hvd.rank(), cache_file_name=cache_file_name)

    columns = ["input_ids", "token_type_ids", "attention_mask"]
    dset.set_format("tensorflow", columns=columns)
    return dset


# 20 milliseconds for WikiText-2
# 20 milliseconds for WikiText-103
# Seems to be loading lazily!
def get_tf_lazy_dataset(nlp_dataset):
    logger.info("Creating gen_dataset from generator")
    output_types = {
        "input_ids": tf.int64,
        "token_type_ids": tf.int64,
        "attention_mask": tf.int64,
    }

    def nlp_dataset_gen():
        for i in range(len(nlp_dataset)):
            yield nlp_dataset[i]

    # from_generator() is bottlenecked by GIL.
    # See https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator
    # So we parallelize it across CPU cores with interleave().
    # https://stackoverflow.com/questions/52179857/parallelize-tf-from-generator-using-tf-contrib-data-parallel-interleave
    tf_dataset = tf.data.Dataset.range(10).interleave(
        lambda idx: tf.data.Dataset.from_generator(nlp_dataset_gen, output_types=output_types),
        cycle_length=10,
    )
    buffer_size = 1000
    # tf_dataset = tf_dataset.shard(hvd.size(), hvd.rank())
    tf_dataset = tf_dataset.repeat()
    tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    tf_dataset = tf_dataset.batch(train_args.per_gpu_batch_size, drop_remainder=True)
    tf_dataset = tf_dataset.prefetch(buffer_size=8)
    logger.info("Finished creating gen_dataset from generator")
    return tf_dataset
