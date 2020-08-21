"""
Example usage:
```bash
python -m common.preprocess --dataset=wikitext-2 --shards=1 --processes=1 --cache_dir=/fsx/data_arrow/wikitext-2 --tfrecords_dir=/fsx/data_tfrecords/wikitext-2_512seq
python -m common.preprocess --dataset=wikibooks --shards=2048 --processes=64 --cache_dir=/fsx/data_arrow/wikibooks --tfrecords_dir=/fsx/data_tfrecords/wikibooks_512seq
```

Inspiration from https://github.com/google-research/electra/blob/master/build_pretraining_dataset.py
25 seconds for WikiText-2 (2M tokens, 84k sentences)
40 minutes for WikiText-103 (103M tokens, 4M sentences)
?? minutes for Wikipedia (2500 tokens, 143M sentences)
?? minutes for BookCorpus (800M tokens, 69M sentences)

The steps are:
1) Download data
2) Filter empty lines (112k it/s)
3) Replace newlines with space (121k it/s)
4) Split on periods into sentences (66k it/s)
5) Pre-tokenize sentences (12k it/s, 3hrs on Wikipedia) -- can be serialized
6) Create examples (24k it/s, 3hrs on Wikipedia) -- can be serialized
7) Convert example tokens into ids (0.15k it/s) -> because of casting ndarray to list? -- can be serialized
8) Export to TFRecords


To directly inspect a TFRecord without knowing the spec:
tfds = tf.data.TFRecordDataset(filenames=[filename])
for batch in tfds.take(1):
    example_proto = tf.train.Example.FromString(batch.numpy())

To attempt loading in a VarLenFeature to see if you didn't serialize everything the same length:
features = {
    "input_ids": tf.io.VarLenFeature(tf.int64),
    "token_type_ids": tf.io.VarLenFeature(tf.int64),
    "attention_mask": tf.io.VarLenFeature(tf.int64),
}
"""

import argparse
import multiprocessing
import os
import random
import sys
import time
from functools import partial
from typing import List

import nlp
import tensorflow as tf
from transformers import BertTokenizerFast

from common.datasets import get_dataset_from_tfrecords

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument(
    "--dataset",
    choices=["wikitext-2", "wikitext-103", "wikipedia", "bookcorpus", "wikibooks", "c4"],
)
parser.add_argument("--cache_dir", default="/tmp/data_arrow")
parser.add_argument("--shards", type=int, default=1)
parser.add_argument("--processes", type=int, default=1)  # 64 processes is a sweet spot on p3dn
parser.add_argument("--tfrecords_dir", default="/tmp/data_tfrecords")
parser.add_argument("--skip_load_from_cache_file", action="store_true")
parser.add_argument("--skip_tfrecords", action="store_true")
args = parser.parse_args()

FILTER_CACHE = "filterlines.arrow"
NEWLINES_CACHE = "replacenewlines.arrow"
SENTENCES_CACHE = "sentences.arrow"
PRETOKENIZED_SENTENCES_CACHE = "pretokenized_sentences.arrow"
EXAMPLES_CACHE = f"examples_{args.max_seq_length}seq.arrow"
EXAMPLE_IDS_CACHE = f"example_ids_{args.max_seq_length}seq.arrow"

load_from_cache_file = not args.skip_load_from_cache_file

assert (
    args.dataset in args.cache_dir
), f"Dataset name '{args.dataset}' should be part of the directory name '{args.cache_dir}', don't mix datasets!"
assert (
    args.skip_tfrecords or args.dataset in args.tfrecords_dir
), f"Dataset name '{args.dataset}' should be part of the TFRecords directory name '{args.tfrecords_dir}', don't mix datasets!"
assert (
    args.skip_tfrecords or str(args.max_seq_length) in args.tfrecords_dir
), f"Sequence length '{args.max_seq_length}' should be part of the TFRecords directory name '{args.tfrecords_dir}', don't mix datasets!"

if not os.path.exists(args.cache_dir):
    os.makedirs(args.cache_dir, exist_ok=True)
if not args.skip_tfrecords and not os.path.exists(args.tfrecords_dir):
    os.makedirs(args.tfrecords_dir, exist_ok=True)

start_time = time.perf_counter()

print(f"Loading dataset: {args.dataset}")
if args.dataset.startswith("wikitext"):
    dset = nlp.load_dataset(
        "wikitext", f"{args.dataset}-raw-v1", split="train", cache_dir=args.cache_dir
    )
elif args.dataset == "wikipedia":
    dset = nlp.load_dataset("wikipedia", "20200501.en", split="train", cache_dir=args.cache_dir)
    dset.drop(columns=["title"])
    dset.features.pop("title")
elif args.dataset == "bookcorpus":
    dset = nlp.load_dataset("bookcorpus", split="train", cache_dir=args.cache_dir)
elif args.dataset == "wikibooks":
    dset_wikipedia = nlp.load_dataset(
        "wikipedia", "20200501.en", split="train", cache_dir=args.cache_dir
    )
    dset_wikipedia.drop(columns=["title"])
    dset_wikipedia.features.pop("title")
    dset_books = nlp.load_dataset("bookcorpus", split="train", cache_dir=args.cache_dir)
    # Cast schemas, since one is nullable and one is not
    dset_wikipedia._data = dset_wikipedia.data.cast(dset_books._data.schema)
    dset = nlp.concatenate_datasets([dset_wikipedia, dset_books])
elif args.dataset == "c4":
    dset = nlp.load_dataset("c4", "en", cache_dir=args.cache_dir)
    assert False, "This dataset must be preprocessed beforehand"
else:
    assert False
print("Loaded dataset:", dset, dset[0])
assert dset.column_names == ["text"], "Dataset should have exactly one 'text' column"

print("Filtering empty lines")
dset = dset.filter(
    lambda ex: len(ex["text"]) > 0,
    cache_file_name=os.path.join(args.cache_dir, FILTER_CACHE),
    load_from_cache_file=load_from_cache_file,
)
print("Filtered empty lines:", dset, dset[0])
print("Replacing newlines with space")
dset = dset.map(
    lambda batch: {"text": [text.strip().replace("\n", " ") for text in batch["text"]]},
    batched=True,
    cache_file_name=os.path.join(args.cache_dir, NEWLINES_CACHE),
    load_from_cache_file=load_from_cache_file,
)
print("Replaced newlines with space:", dset, dset[0])


def split_into_sentences(batch):
    """ Split into sentences using the '.' separator. Not perfect, converts

    Senjō no Valkyria 3 : Unrecorded Chronicles (
    Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the
    Battlefield 3 ) , commonly referred to as Valkyria
    Chronicles III outside Japan , is a tactical role
    @-@ playing video game developed by Sega and
    Media.Vision for the PlayStation Portable .

    into three sentences when it really is one. But works pretty well.
    """
    sentences = []
    for ex in batch["text"]:
        batch_sentences = [sentence + "." for sentence in ex.split(".")]
        batch_sentences = batch_sentences[:-1]
        sentences.extend(batch_sentences)
    return {"sentences": sentences}


print("Splitting into sentences")
dset = dset.map(
    split_into_sentences,
    batched=True,
    remove_columns=["text"],
    cache_file_name=os.path.join(args.cache_dir, SENTENCES_CACHE),
    load_from_cache_file=load_from_cache_file,
)
print("Split into sentences:", dset, dset[0])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def pretokenize(batch):
    """ Tokenize via list comprehension in Rust. """
    encodings: List["Encoding"] = tokenizer._tokenizer.encode_batch(
        batch["sentences"], add_special_tokens=False
    )
    tokens: List[str] = [encoding.tokens for encoding in encodings]
    return {"tokens": tokens}


# dset = dset.select(np.arange(0, 60000))
print("Pre-tokenizing sentences:")
dset = dset.map(
    pretokenize,
    batched=True,
    remove_columns=["sentences"],
    cache_file_name=os.path.join(args.cache_dir, PRETOKENIZED_SENTENCES_CACHE),
    load_from_cache_file=load_from_cache_file,
)
print("Pre-tokenized sentences:", dset, dset[0])


def create_examples(batch, max_length):
    """Creates a pre-training example from the current list of sentences."""
    target_length = max_length
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
        first_segment_target_length = 100000
    else:
        # -3 due to not yet having [CLS]/[SEP] tokens in the input text
        first_segment_target_length = (target_length - 3) // 2

    first_segment, second_segment = [], []
    examples = []
    for sentence in batch["tokens"]:
        # the sentence goes to the first segment if (1) the first segment is
        # empty, (2) the sentence doesn't put the first segment over length or
        # (3) 50% of the time when it does put the first segment over length
        if (
            len(first_segment) == 0
            or len(first_segment) + len(sentence) < first_segment_target_length
            or (
                len(second_segment) == 0
                and len(first_segment) < first_segment_target_length
                and random.random() < 0.5
            )
        ):
            first_segment += list(sentence)
        else:
            second_segment += list(sentence)
            if len(first_segment) + len(second_segment) >= target_length:
                # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
                first_segment = first_segment[: max_length - 2]
                second_segment = second_segment[: max(0, max_length - len(first_segment) - 3)]
                example = ["[CLS]"] + first_segment + ["[SEP]"] + second_segment + ["[SEP]"]
                examples.append(example)
                first_segment, second_segment = [], []

                if random.random() < 0.05:
                    target_length = random.randint(5, max_length)
                else:
                    target_length = max_length

    # This last one may be a little short, but it's necessary to always return something from the function
    # for the function inspection that only passes two sentences.
    examples.append(["[CLS]"] + first_segment + ["[SEP]"] + second_segment + ["[SEP]"])

    return {"examples": examples}


print("Creating examples")
dset = dset.map(
    partial(create_examples, max_length=args.max_seq_length),
    batched=True,
    remove_columns=["tokens"],
    cache_file_name=os.path.join(args.cache_dir, EXAMPLES_CACHE),
    load_from_cache_file=load_from_cache_file,
)
print("Created examples:", dset, dset[0])
# WARNING: Some of these examples are shorter than 512 sequence length.
# View with [len(ex["examples"]) for ex in dset]

# This method is very slow (0.15 it/s, so 0.15k examples/sec
# Improvement tracked in https://github.com/huggingface/transformers/issues/5729
print(f"Padding, truncating, and encoding examples into ids. num_processes={args.processes}")


def tokenizer_batch(batch, tokenizer):
    # This must be defined in __main__ for serialization
    return tokenizer(
        batch["examples"],
        add_special_tokens=False,
        is_pretokenized=True,
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_length,
    )


def shard_and_map(index, filename, num_shards, function, **kwargs):
    print(f"Sharding on process {index}")
    shard = nlp.Dataset.from_file(filename).shard(
        num_shards, index, contiguous=True, load_from_cache_file=load_from_cache_file
    )
    print(f"Done sharding on process {index}. Mapping the shard")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return shard.map(partial(function, tokenizer=tokenizer), **kwargs)


def multiprocess_map(dset, num_processes, function, **kwargs):
    with multiprocessing.Pool(processes=num_processes) as pool:
        shards = pool.map(
            partial(
                shard_and_map,
                filename=dset._data_files[0]["filename"],
                num_shards=num_processes,
                function=function,
                **kwargs,
            ),
            range(num_processes),
        )
    return nlp.concatenate_datasets(shards)


dset = multiprocess_map(
    dset=dset,
    num_processes=args.processes,
    function=tokenizer_batch,
    batched=True,
    remove_columns=["examples"],
    cache_file_name=os.path.join(args.cache_dir, EXAMPLE_IDS_CACHE),
    load_from_cache_file=load_from_cache_file,
)
print("Padded, truncated, and encoded examples into ids:", dset, dset[0])
# dset = nlp.Dataset.from_file(cache_file)

if args.skip_tfrecords:
    sys.exit()

### Export to sharded TFRecords ###

tfrecord_files = [
    os.path.join(args.tfrecords_dir, f"{args.dataset}_shard_{i}.tfrecord")
    for i in range(args.shards)
]


def shard_and_export(index):
    dset_shard = dset.shard(num_shards=args.shards, index=index, contiguous=False)
    dset_shard.set_format("numpy")
    dset_shard.export(tfrecord_files[index])


# Beware of TensorFlow + multiprocessing. Ensure there are no visible GPUs so everything happens on CPU.
with multiprocessing.Pool(processes=args.processes) as pool:
    pool.map(shard_and_export, range(args.shards))

### Now read in a TFRecord to ensure exporting happened correctly ###

name_to_features = {
    "input_ids": tf.io.FixedLenFeature([args.max_seq_length], tf.int64),  # corresponds to input_ids
    "token_type_ids": tf.io.FixedLenFeature(
        [args.max_seq_length], tf.int64
    ),  # corresponds to token_type_ids
    "attention_mask": tf.io.FixedLenFeature(
        [args.max_seq_length], tf.int64,
    ),  # corresponds to attention_mask
}

tfds = get_dataset_from_tfrecords(
    model_type="electra",
    filenames=tfrecord_files,
    max_seq_length=args.max_seq_length,
    per_gpu_batch_size=4,
    shard=False,
)
for batch in tfds.take(1):
    print(batch)

elapsed = time.perf_counter() - start_time
print(f"Total processing time: {elapsed:.3f} seconds")
