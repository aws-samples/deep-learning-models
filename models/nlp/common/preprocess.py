import time

import nlp
import numpy as np
import tensorflow as tf

dset = nlp.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dset = dset.filter(lambda ex: len(ex["text"]) > 0)
bsz = 1024
n_batches = 100


def single_item_gen():
    for i in range(len(dset)):
        yield dset[i]


def sequential_batch_gen():
    for i in range(0, len(dset), bsz):
        yield dset[i : i + bsz]


def random_batch_gen():
    for i in range(len(dset)):
        indices = list(np.random.randint(len(dset), size=(bsz,)))
        yield dset[indices]


output_types = {"text": tf.string}
single_item = tf.data.Dataset.from_generator(single_item_gen, output_types=output_types).batch(bsz)
interleaved = tf.data.Dataset.range(10).interleave(
    lambda idx: tf.data.Dataset.from_generator(single_item_gen, output_types=output_types),
    cycle_length=10,
)
sequential_batch = tf.data.Dataset.from_generator(sequential_batch_gen, output_types=output_types)
random_batch = tf.data.Dataset.from_generator(random_batch_gen, output_types=output_types)


def iterate(tf_dset):
    start = time.perf_counter()
    for i, batch in enumerate(tf_dset.take(n_batches)):
        pass
    elapsed = time.perf_counter() - start
    print(f"{tf_dset} took {elapsed:.3f} secs")


iterate(single_item)
iterate(interleaved)
iterate(sequential_batch)
iterate(random_batch)
