"""
Batch sizes: 32 = 5GB memory, 128 = 17GB

The "read -1 expected ..." errors are harmless and come from Docker. See https://github.com/horovod/horovod/issues/503
Running Docker in privileged mode (docker run --privileged) solves the issue.

Dataset handling: Lots of empty lines, use dataset.filter() to eliminate those.
For now, just grab one sentence.
TODO: Combine two segments into a single example. https://github.com/google-research/electra/blob/master/build_pretraining_dataset.py
TODO: Add zero-padding for shorter sequences

nlp feature request: Select from dataset with arbitrary slices
`nlp` package tutorial: https://colab.research.google.com/github/huggingface/nlp/blob/master/notebooks/Overview.ipynb
"""

import datetime
import logging
import time

import numpy as np
import tensorflow as tf
from nlp import load_dataset
from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    ElectraTokenizerFast,
    HfArgumentParser,
    TFElectraForMaskedLM,
    TFElectraForPreTraining,
)

from common.arguments import (
    DataTrainingArguments,
    LoggingArguments,
    ModelArguments,
    TrainingArguments,
)
from common.optimizers import get_adamw_optimizer
from common.utils import TqdmLoggingHandler, is_wandb_available
from electra.utils import colorize_dis, colorize_gen

# See https://github.com/huggingface/transformers/issues/3782; this import must come last
import horovod.tensorflow as hvd  # isort:skip

if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)


def log_example(tokenizer, ids, masked_ids, mask, gen_ids, dis_preds):
    logger.info(f"ORIGINAL:      '{tokenizer.decode(ids[0].numpy())}'")
    logger.info(f"MASKED:        '{tokenizer.decode(masked_ids[0].numpy())}'")
    logger.info(f"GENERATOR:     '{colorize_gen(tokenizer, ids[0], gen_ids[0], mask[0])}'")
    logger.info(f"DISCRIMINATOR: '{colorize_dis(tokenizer, gen_ids[0], dis_preds[0])}'")


# TODO: Limit code duplication between train_step and val_step.
# Abstracting logic out into another function gets messy because of tf.function wrapping & caching,
# long lists of parameters to pass in and long lists of return values.
# TODO: Re-add validation step


@tf.function
def train_step(optimizer, gen, dis, ids, masked_ids, attention_mask, corruption_mask):
    """
    Attention mask refers to padding tokens.
        1 is a real token, 0 is a padding token.
    Corruption mask refers to which tokens are replaced by the generator.
        1 is a corrupted (replaced) token, 0 is an original token.
    tf.boolean_mask([[1,2], [3,4], [5,6]], [True, False, True]) -> [[1,2], [5,6]]
    Id-to-token reference:
        0: PAD
        103: MASK
    """
    with tf.GradientTape() as tape:
        ids = tf.cast(ids, tf.int64)
        corruption_mask = tf.cast(corruption_mask, tf.int64)

        # Generator loss
        (gen_logits,) = gen(
            {"input_ids": masked_ids, "attention_mask": attention_mask}
        )  # [bsz, seq_len, vocab_size]
        gen_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=tf.boolean_mask(ids, corruption_mask),  # [bsz * n_masks, vocab_size]
            y_pred=tf.boolean_mask(gen_logits, corruption_mask),  # [bsz * n_masks]
            from_logits=True,
        )  # [bsz * n_masks]
        gen_loss = tf.reduce_mean(gen_loss)  # [1]

        # Generator accuracy
        # argmax returns tf.int64 by default
        adv_ids = tf.argmax(gen_logits, axis=-1, output_type=ids.dtype)  # [bsz, seq_len]
        ids_equal = tf.cast(adv_ids == ids, dtype=tf.int64)  # [bsz, seq_len]
        gen_correct = tf.boolean_mask(ids_equal, corruption_mask)  # [bsz * n_masks]
        gen_acc = tf.reduce_mean(tf.cast(gen_correct, dtype=tf.float32))  # [1]

        # Discriminator loss
        gen_ids = corruption_mask * adv_ids + (1 - corruption_mask) * ids  # [bsz, seq_len]
        (dis_logits,) = dis(
            {"input_ids": gen_ids, "attention_mask": attention_mask}
        )  # [bsz, seq_len]
        # If generator generates correct token, invert the loss
        is_corrupted = tf.cast(gen_ids != ids, tf.int64)
        dis_loss = tf.keras.losses.binary_crossentropy(
            y_true=tf.boolean_mask(is_corrupted, attention_mask),
            y_pred=tf.boolean_mask(dis_logits, attention_mask),
            from_logits=True,
        )
        dis_loss = tf.reduce_mean(dis_loss)

        # Discriminator accuracy
        # TODO: Check that accuracy_mask is different
        dis_probs = tf.math.sigmoid(dis_logits)  # [bsz, seq_len]
        dis_preds = tf.cast(dis_probs > 0.5, dtype=corruption_mask.dtype)  # [bsz, seq_len] (bool)
        dis_acc = tf.reduce_mean(
            tf.cast(tf.cast(dis_preds, tf.bool) == (gen_ids != ids), dtype=tf.float32)
        )  # gen_ids != ids is corrupted

        # Generator is 30,000-way classification loss, while discriminator is binary classification.
        lmbda = 50
        loss = gen_loss + lmbda * dis_loss

    vars = gen.trainable_variables + dis.trainable_variables
    grads = tape.gradient(loss, vars)
    grads = [
        hvd.allreduce(grad, compression=hvd.Compression.fp16) if grad is not None else None
        for grad in grads
    ]
    optimizer.apply_gradients(zip(grads, vars))

    return loss, gen_loss, dis_loss, gen_acc, dis_acc, gen_ids, dis_preds


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments)
    )
    (
        model_args,
        data_args,
        train_args,
        log_args,
        remaining_strings,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    # SageMaker may have some extra strings. TODO: Test this on SM.
    assert len(remaining_strings) == 0, f"The args {remaining_strings} could not be parsed."

    hvd.init()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    if train_args.eager == "true":
        tf.config.experimental_run_functions_eagerly(True)

    tokenizer = ElectraTokenizerFast.from_pretrained("bert-base-uncased")

    gen_config = ElectraConfig.from_pretrained("google/electra-small-generator")
    dis_config = ElectraConfig.from_pretrained("google/electra-small-discriminator")

    gen = TFElectraForMaskedLM(config=gen_config)
    dis = TFElectraForPreTraining(config=dis_config)
    optimizer = get_adamw_optimizer(train_args)

    start_time = time.perf_counter()

    if hvd.rank() == 0:
        # Logging should only happen on a single process
        # https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
        level = logging.INFO
        format = "%(asctime)-15s %(name)-12s: %(levelname)-8s %(message)s"
        handlers = [
            TqdmLoggingHandler(),
        ]
        logging.basicConfig(level=level, format=format, handlers=handlers)
        wandb_run_name = None

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if log_args.run_name is None:
            metadata = (
                f"electra-{hvd.size()}gpus"
                f"-{train_args.per_gpu_batch_size * hvd.size() * train_args.gradient_accumulation_steps}globalbatch"
                f"-{train_args.total_steps}steps"
            )
            run_name = (
                f"{current_time}-{metadata}-{train_args.name if train_args.name else 'unnamed'}"
            )
        else:
            run_name = f"{current_time}-{log_args.run_name}"

    def generate_corruption_mask(ids, attention_mask):
        mask = (
            tf.cast(tf.random.uniform(shape=ids.shape) > 0.85, dtype=attention_mask.dtype)
            * attention_mask
        )
        return mask

    def mask_ids(ids, corruption_mask, mask_id):
        return tf.where(tf.cast(corruption_mask, tf.bool), mask_id, ids)

    def remove_none_values(example):
        return example["text"] != ""

    def tokenize(example):
        # return tokenizer.batch_encode_plus(example['text'])
        return tokenizer(
            example["text"], padding=True, truncation=True, max_length=data_args.max_seq_length
        )

    portion = 100 / hvd.size()
    start = int(hvd.rank() * portion)
    end = int(start + portion)
    dataset = "wikitext-2"  # or wikitext-103
    train_dataset = load_dataset("wikitext", f"{dataset}-raw-v1", split=f"train[{start}%:{end}%]")
    train_dataset = train_dataset.filter(remove_none_values)
    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        cache_file_name=f"/fsx/{dataset}.cache",
        load_from_cache_file=False,
    )
    # Or load it in:
    # train_dataset = Dataset.from_file(f"/fsx/{dataset}.cache")

    columns = ["input_ids", "token_type_ids", "attention_mask"]
    train_dataset.set_format("tensorflow", columns=columns)
    batch = train_dataset[:5]["input_ids"]  # RaggedTensor
    batch = batch.to_tensor(default_value=0, shape=[None, data_args.max_seq_length])  # Tensor
    gen(batch)

    # Is this lazy evaluation?
    # TODO: Convert to from_generator()
    logger.info("Creating tf_dataset from tensor_slices")  # Takes 18 secs for WikiText-2
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        ({x: train_dataset[x].to_tensor() for x in columns})
    ).batch(train_args.per_gpu_batch_size)
    logger.info("Finished creating tf_dataset from tensor_slices")

    wandb_run_name = None

    tf_dataset_iter = iter(tf_dataset)
    for step in range(1, train_args.total_steps + 1):
        batch_dict = next(tf_dataset_iter)
        ids = batch_dict["input_ids"]
        attention_mask = batch_dict["attention_mask"]
        corruption_mask = generate_corruption_mask(ids=ids, attention_mask=attention_mask)
        masked_ids = mask_ids(
            ids=ids, corruption_mask=corruption_mask, mask_id=tokenizer.mask_token_id
        )

        loss, gen_loss, dis_loss, gen_acc, dis_acc, gen_ids, dis_preds = train_step(
            optimizer=optimizer,
            gen=gen,
            dis=dis,
            ids=ids,
            masked_ids=masked_ids,
            attention_mask=attention_mask,
            corruption_mask=corruption_mask,
        )

        if step == 1:
            # Horovod broadcast
            hvd.broadcast_variables(dis.variables, root_rank=0)
            hvd.broadcast_variables(gen.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        if hvd.rank() == 0:
            is_final_step = step >= train_args.total_steps
            do_log = step % log_args.log_frequency == 0
            do_checkpoint = (step > 0) and (
                (step % log_args.checkpoint_frequency == 0) or is_final_step
            )
            do_validation = False  # step % log_args.validation_frequency == 0

            if do_log:
                elapsed_time = time.perf_counter() - start_time  # Off for first log
                it_s = log_args.log_frequency / elapsed_time
                start_time = time.perf_counter()
                log_example(tokenizer, ids, masked_ids, corruption_mask, gen_ids, dis_preds)
                description = f"Step {step} -- gen_loss: {gen_loss:.3f}, dis_loss: {dis_loss:.3f}, gen_acc: {gen_acc:.3f}, dis_acc: {dis_acc:.3f}, it/s: {it_s:.3f}\n"
                logger.info(description)

            if do_validation:
                # TODO: Re-implement validation, but with less code duplication
                # val_ids = get_batch_ids(dataset=train_dataset, bsz=bsz, seq_len=seq_len)
                # val_masked_ids, val_mask = mask_ids(val_ids)
                # (
                #     val_loss,
                #     val_gen_loss,
                #     val_dis_loss,
                #     val_gen_acc,
                #     val_dis_acc,
                #     val_gen_ids,
                #     val_dis_preds,
                # ) = val_step(
                #     gen=gen, dis=dis, ids=val_ids, masked_ids=val_masked_ids, mask=val_mask
                # )
                # log_example(
                #     tokenizer, val_ids, val_masked_ids, val_mask, val_gen_ids, val_dis_preds
                # )
                # description = f"VALIDATION, Step {step} -- val_gen_loss: {val_gen_loss:.3f}, val_dis_loss: {val_dis_loss:.3f}, val_gen_acc: {val_gen_acc:.3f}, val_dis_acc: {val_dis_acc:.3f}\n"
                # logger.info(description)
                pass

            train_metrics = {
                "train/loss": loss,
                "train/gen_loss": gen_loss,
                "train/dis_loss": dis_loss,
                "train/gen_acc": gen_acc,
                "train/dis_acc": dis_acc,
            }
            all_metrics = {**train_metrics}
            if do_validation:
                val_metrics = {
                    "val/loss": val_loss,
                    "val/gen_loss": val_gen_loss,
                    "val/dis_loss": val_dis_loss,
                    "val/gen_acc": val_gen_acc,
                    "val/dis_acc": val_dis_acc,
                }
                all_metrics = {**all_metrics, **val_metrics}
            if do_log:
                all_metrics = {"it_s": it_s, **all_metrics}

            if is_wandb_available():
                if wandb_run_name is None:
                    config = {
                        "global_batch_size": hvd.size() * train_args.per_gpu_batch_size,
                        "per_gpu_batch_size": train_args.per_gpu_batch_size,
                        "max_seq_length": data_args.max_seq_length,
                    }
                    wandb.init(config=config, project="electra")
                    wandb.run.save()
                    wandb_run_name = wandb.run.name
                wandb.log({"step": step, **all_metrics})

            if do_checkpoint:
                dis_model_ckpt = f"{data_args.fsx_prefix}/checkpoints/electra/discriminator-{run_name}-step{step}.ckpt"
                gen_model_ckpt = f"{data_args.fsx_prefix}/checkpoints/electra/generator-{run_name}-step{step}.ckpt"
                optimizer_ckpt = f"{data_args.fsx_prefix}/checkpoints/electra/optimizer-{run_name}-step{step}.npy"
                logger.info(
                    f"Saving discriminator model at {dis_model_ckpt}, generator model at {gen_model_ckpt}, optimizer at {optimizer_ckpt}"
                )
                dis.save_weights(dis_model_ckpt)
                gen.save_weights(gen_model_ckpt)
                np.save(optimizer_ckpt, optimizer.get_weights())


if __name__ == "__main__":
    main()
