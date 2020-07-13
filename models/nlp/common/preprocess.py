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
from transformers import BertTokenizerFast

load_from_cache_file = True

dset = nlp.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
print("Loaded dataset:", dset, dset[0])
dset = dset.filter(lambda ex: len(ex["text"]) > 0)
print("Filtered empty lines:", dset, dset[0])
dset = dset.map(
    lambda batch: {"text": [text.strip().replace("\n", " ") for text in batch["text"]]},
    batched=True,
)
print("Replaced newlines with space:", dset, dset[0])


def join_documents(batch):
    """ Each document starts with a `= Title =`, and subheadings have two/three equals signs. """


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


dset = dset.map(
    split_into_sentences,
    batched=True,
    remove_columns=["text"],
    load_from_cache_file=load_from_cache_file,
)
print("Split into sentences:", dset, dset[0])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def tokenize(batch):

    return {"tokens": [tokenizer.tokenize(example) for example in batch["sentences"]]}


dset = dset.map(tokenize, batched=True, remove_columns=["sentences"])
print("Tokenized sentences:", dset, dset[0])

# def tokens_to_ids(first_segment, second_segment):
#     sequence = ["[CLS]"] + first_segment + ["[SEP]"] + second_segment + ["[SEP]"]
#     return tokenizer.convert_tokens_to_ids(sequence)


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


dset = dset.map(
    partial(create_examples, max_length=512),
    batched=True,
    remove_columns=["tokens"],
    load_from_cache_file=load_from_cache_file,
)
print("Created examples:", dset, dset[0])

# TODO: Just use the normal __call__ so it will do the padding for us.
def tokenize_from_pretokenized(ex):
    string = tokenizer.decode(tokenizer.convert_tokens_to_ids(ex["examples"]))
    return tokenizer(string, padding=True, truncation=True, max_length=512)


dset = dset.map(
    tokenize_from_pretokenized,
    remove_columns=["examples"],
    cache_file_name="/Users/nieljare/Desktop/wikitext103-encoded.cache",
    load_from_cache_file=load_from_cache_file,
)
print("Padded, truncated, and encoded examples into ids:", dset, dset[0])

breakpoint()
