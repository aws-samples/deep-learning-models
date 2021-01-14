"""
Launch with
`python -m t5.mini_t5`
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
import tqdm
from transformers import T5Tokenizer, TFT5ForConditionalGeneration


def convert_to_json(text):
    return json.dumps({"text": text})


def get_dataset():
    out_filename = "/fsx/t5-data/data/c4-train.txt-00001-of-01024"
    filenames = [
        "/fsx/t5_pretraining/tensorflow_datasets/c4/en/2.3.1/c4-train.tfrecord-00001-of-01024"
    ]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    feature_description = {
        "text": tf.io.VarLenFeature(tf.string),
        "content-length": tf.io.VarLenFeature(tf.string),
        "content-type": tf.io.VarLenFeature(tf.string),
        "timestamp": tf.io.VarLenFeature(tf.string),
        "url": tf.io.VarLenFeature(tf.string),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    if os.path.isfile(out_filename):
        os.remove(out_filename)
    for parsed_record in tqdm.tqdm(parsed_dataset):
        text: str = parsed_record["text"].values[0].numpy().decode("utf-8")
        json_text = convert_to_json(text)
        # print(repr(parsed_record))
        # print(parsed_record["text"].values)
        with open(out_filename, "a") as outfile:
            outfile.write(json_text + "\n")

    foo = 3
    # {
    #         "text": "byte",
    #         "content-length": "byte",
    #         "content-type": "byte",
    #         "timestamp": "byte",
    #         "url": "byte"}


def gen_synthetic_batch(batch_size: int, sequence_length: int):
    data = np.zeros((batch_size, sequence_length))
    tensor = tf.convert_to_tensor(data, dtype=tf.int64)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
        default="t5-small",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=128)

    args = parser.parse_args()

    print(
        f"Training {args.model_name} for {args.steps} steps with batch size {args.batch_size} and sequence length {args.sequence_length}"
    )

    model = TFT5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    optimizer = tf.keras.optimizers.Adam(0.001)

    log_frequency = 10
    start_time = time.perf_counter()
    for i in range(args.steps):
        # dataset = get_dataset()
        inputs = gen_synthetic_batch(
            batch_size=args.batch_size, sequence_length=args.sequence_length
        )  # [batch_size, sequence_length]

        # Start with text
        text = "Hello, my name is Jared."
        # Now tokenize
        tokenized = tokenizer(text)  # dict with keys "input_ids", "attention_mask"

        # dataset = get_dataset()
        model.generate(
            tokenizer(text, return_tensors="tf")["input_ids"],
            max_length=50,
            num_beams=5,
            early_stopping=True,
        )

        with tf.GradientTape() as tape:
            outputs = model(inputs, decoder_input_ids=inputs)
            last_hidden_states = (
                outputs.encoder_last_hidden_state
            )  # [batch_size, sequence_length, embedding_size]
            loss = tf.reduce_mean(last_hidden_states ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if i % log_frequency == 0:
            elapsed_time = time.perf_counter() - start_time
            it_s = log_frequency / elapsed_time
            start_time = time.perf_counter()
            print(f"Step {i}, Loss: {loss.numpy():.3f}, It/s: {it_s:.3f}")


if __name__ == "__main__":
    main()
    # get_dataset()
