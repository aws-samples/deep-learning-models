"""
Inspiration from https://github.com/google-research/electra/blob/master/build_pretraining_dataset.py

TODO: Parse into documents.
I might accidentally cross document boundaries with this technique.
How many examples should I generate?
Assert we don't generate any examples less than 512
"""

import random
from functools import partial

import nlp
import numpy as np

dset = nlp.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
print(dset, dset[0])
dset = dset.filter(lambda ex: len(ex["text"]) > 0)
print(dset)
dset = dset.map(
    lambda batch: {"text": [text.strip().replace("\n", " ") for text in batch["text"]]},
    batched=True,
)
print(dset, dset[0])


def join_documents(batch):
    """ Each document starts with a `= Title =`, and subheadings have two/three equals signs. """


def split_sentences(batch):
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


dset = dset.map(split_sentences, batched=True, remove_columns=["text"], load_from_cache_file=False)
print(dset, dset[0])

# def group_sentences(batch, target_len: int):
#     glob = []
#     cur_str = ""
#     for ex in batch["sentences"]:
#         cur_str += ex
#         if len(cur_str) > target_len:
#             glob.append(cur_str)
#             cur_str = ""
#     glob.append(cur_str)
#     return {"grouped_sentences": glob}

# dset = dset.map(partial(group_sentences, target_len=5000), batched=True, remove_columns=["sentences"], load_from_cache_file=False)
# print(dset, dset[0])


def create_example(sentences, max_length, target_length):
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
        first_segment_target_length = 100000
    else:
        # -3 due to not yet having [CLS]/[SEP] tokens in the input text
        first_segment_target_length = (target_length - 3) // 2

    first_segment = []
    second_segment = []
    for sentence in sentences:
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
            first_segment += sentence
        else:
            second_segment += sentence

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[: max_length - 2]
    second_segment = second_segment[: max(0, max_length - len(first_segment) - 3)]

    # prepare to start building the next example
    self._current_sentences = []
    self._current_length = 0
    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
        target_length = random.randint(5, max_length)
    else:
        target_length = max_length

    return f"[CLS]{first_segment}[SEP]{second_segment}[SEP]"


def create_examples(batch, n_examples_per_batch: int, max_seq_len: int):
    examples = []
    for _ in range(n_examples_per_batch):
        s1, s2 = np.random.randint(len(batch["sentences"]), size=(2,))
        text1, text2 = batch["sentences"][s1], batch["sentences"][s2]
        i1 = np.random.randint(len(text1)) - max_seq_len
        i2 = np.random.randint(len(text2)) - max_seq_len
        example = f"[CLS]{text1[i1:i1+max_seq_len]}[SEP]{text2[i2:i2+max_seq_len]}[SEP]"
        examples.append(example)
    return {"examples": examples}


def create_examples_from_sentences(batch, n_examples_per_batch: int, max_seq_len: int):
    examples = []
    for _ in range(n_examples_per_batch):
        pass


dset = dset.map(
    partial(create_examples, n_examples_per_batch=1, max_seq_len=128),
    batched=True,
    remove_columns=["sentences"],
    load_from_cache_file=False,
)
print(dset, dset[0])


# breakpoint()
