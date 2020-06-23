"""
Batch sizes: 32 = 5GB memory, 128 = 17GB

What's the best way to develop a new pretraining script?

Dynamic masking straight from text.
Abstract out the gradient accumulation functionality. Tracking loss, acc variables within the accumulator rather than outside.
Incorporate the new transformers version. Be willing to lose my current work.

# TODO: Should we include special tokens? <BOS>, <EOS>.
# TODO: Weight sharing between generator and discriminator, only token embeddings.

The "read -1 expected ..." errors are harmless and come from Docker. See https://github.com/horovod/horovod/issues/503
Running Docker in privileged mode (docker run --privileged) solves the issue.
"""

import datetime
import logging
import time
from typing import Tuple

import numpy as np
import tensorflow as tf
from transformers import (
    ElectraConfig,
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


def select_ids(arr, seq_len: int) -> np.array:
    """ Given an array and sequence length, select a subsequence of that length. """
    start = 0 if len(arr) <= seq_len else np.random.randint(0, len(arr) - seq_len)
    return np.array(arr[start : start + seq_len])


def select_batch_ids(arr, bsz: int, seq_len: int) -> np.array:
    """ Select a batch of select_ids(). """
    out = np.zeros(shape=(bsz, seq_len), dtype=int)
    for i in range(bsz):
        out[i] = select_ids(arr, seq_len)
    return out


# Can we pass labels into model.forward? Simplify the forward pass as much as possible.
def val_step(gen, dis, ids, masked_ids, mask):
    # Generator loss
    (gen_logits,) = gen(masked_ids)  # [bsz, seq_len, vocab_size]
    gen_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=tf.boolean_mask(ids, mask),  # [bsz * n_masks, vocab_size]
        y_pred=tf.boolean_mask(gen_logits, mask),  # [bsz * n_masks]
        from_logits=True,
    )  # [bsz * n_masks]
    gen_loss = tf.reduce_mean(gen_loss)  # [1]

    # Generator accuracy
    adv_ids = tf.argmax(gen_logits, axis=-1)  # [bsz, seq_len]
    ids_equal = tf.cast(adv_ids == ids, dtype=tf.int64)  # [bsz, seq_len] (bool)
    gen_correct = tf.boolean_mask(ids_equal, mask)  # [bsz * n_masks]
    gen_acc = tf.reduce_mean(tf.cast(gen_correct, dtype=tf.float32))  # [1]

    # Discriminator loss
    gen_ids = mask * adv_ids + (1 - mask) * ids  # [bsz, seq_len]
    (dis_logits,) = dis(gen_ids)  # [bsz, seq_len]
    # If generator generates correct token, invert the loss
    dis_loss = tf.keras.losses.binary_crossentropy(
        y_true=tf.cast(gen_ids != ids, tf.int64), y_pred=dis_logits, from_logits=True
    )
    dis_loss = tf.reduce_mean(dis_loss)

    # Discriminator accuracy
    dis_probs = tf.math.sigmoid(dis_logits)  # [bsz, seq_len]
    dis_preds = tf.cast(dis_probs > 0.5, dtype=mask.dtype)  # [bsz, seq_len] (bool)
    dis_acc = tf.reduce_mean(
        tf.cast(tf.cast(dis_preds, tf.bool) == (gen_ids != ids), dtype=tf.float32)
    )  # gen_ids != ids is corrupted

    # Generator is 30,000-way classification loss, while discriminator is binary classification.
    lmbda = 50
    loss = gen_loss + lmbda * dis_loss

    return loss, gen_loss, dis_loss, gen_acc, dis_acc, gen_ids, dis_preds


@tf.function
def train_step(optimizer, gen, dis, ids, masked_ids, mask):
    with tf.GradientTape() as tape:
        # Generator loss
        (gen_logits,) = gen(masked_ids)  # [bsz, seq_len, vocab_size]
        gen_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=tf.boolean_mask(ids, mask),  # [bsz * n_masks, vocab_size]
            y_pred=tf.boolean_mask(gen_logits, mask),  # [bsz * n_masks]
            from_logits=True,
        )  # [bsz * n_masks]
        gen_loss = tf.reduce_mean(gen_loss)  # [1]

        # Generator accuracy
        adv_ids = tf.argmax(gen_logits, axis=-1)  # [bsz, seq_len]
        ids_equal = tf.cast(adv_ids == ids, dtype=tf.int64)  # [bsz, seq_len] (bool)
        gen_correct = tf.boolean_mask(ids_equal, mask)  # [bsz * n_masks]
        gen_acc = tf.reduce_mean(tf.cast(gen_correct, dtype=tf.float32))  # [1]

        # Discriminator loss
        gen_ids = mask * adv_ids + (1 - mask) * ids  # [bsz, seq_len]
        (dis_logits,) = dis(gen_ids)  # [bsz, seq_len]
        # If generator generates correct token, invert the loss
        dis_loss = tf.keras.losses.binary_crossentropy(
            y_true=tf.cast(gen_ids != ids, tf.int64), y_pred=dis_logits, from_logits=True
        )
        dis_loss = tf.reduce_mean(dis_loss)

        # Discriminator accuracy
        dis_probs = tf.math.sigmoid(dis_logits)  # [bsz, seq_len]
        dis_preds = tf.cast(dis_probs > 0.5, dtype=mask.dtype)  # [bsz, seq_len] (bool)
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

    # TODO: Should I use bert-base-uncased?
    tokenizer = ElectraTokenizerFast.from_pretrained("bert-base-uncased")

    gen_config = ElectraConfig.from_pretrained("google/electra-small-generator")
    dis_config = ElectraConfig.from_pretrained("google/electra-small-discriminator")

    # gen = TFElectraForMaskedLM.from_pretrained("google/electra-small-generator")
    # dis = TFElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
    gen = TFElectraForMaskedLM(config=gen_config)
    dis = TFElectraForPreTraining(config=dis_config)
    optimizer = get_adamw_optimizer(train_args)

    # Load in WikiText-2.
    # WikiText-2 train contains 2M tokens
    # WikiText-103 train contains 103M tokens
    train_filename = "/fsx/wikitext/wikitext-2-raw/wiki.train.raw"  # Tokenization complete in 25.394 secs, 7.386 secs with fast tokenizer
    val_filename = "/fsx/wikitext/wikitext-2-raw/wiki.valid.raw"
    test_filename = (
        "/fsx/wikitext/wikitext-2-raw/wiki.test.raw"  # Tokenization complete in 2.964 secs,
    )
    # train_filename = "/fsx/wikitext/wikitext-103-raw/wiki.train.raw" # Fast tokenization in 380.913 secs
    start_time = time.perf_counter()

    def tokenize(filename):
        with open(train_filename) as infile:
            text: str = infile.read()
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        return ids

    wiki_ids = tokenize(train_filename)

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
        logger.info(f"Tokenization complete in {time.perf_counter() - start_time:.3f} secs")

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

        val_ids = tokenize(val_filename)

    def mask_ids(ids: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        # Generate a mask.
        # Mask should be a boolean array where 1 represents masked token.
        mask_prob = 0.15
        mask = np.array(np.random.rand(*ids.shape) > 1 - mask_prob, dtype=int)
        tf_mask = tf.constant(mask)
        # Mask the token ids.
        masked_ids = np.where(mask, tokenizer.mask_token_id, ids)
        tf_masked_ids = tf.constant(masked_ids)
        return tf_masked_ids, tf_mask

    wandb_run_name = None
    for step in range(1, train_args.total_steps + 1):
        bsz = train_args.per_gpu_batch_size
        seq_len = data_args.max_seq_length
        ids = tf.constant(
            select_batch_ids(arr=wiki_ids, bsz=bsz, seq_len=seq_len)
        )  # [bsz, seq_len]
        masked_ids, mask = mask_ids(ids)

        loss, gen_loss, dis_loss, gen_acc, dis_acc, gen_ids, dis_preds = train_step(
            optimizer=optimizer, gen=gen, dis=dis, ids=ids, masked_ids=masked_ids, mask=mask
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
            do_validation = step % log_args.validation_frequency == 0

            if do_log:
                elapsed_time = time.perf_counter() - start_time  # Off for first log
                it_s = log_args.log_frequency / elapsed_time
                start_time = time.perf_counter()
                log_example(tokenizer, ids, masked_ids, mask, gen_ids, dis_preds)
                description = f"Step {step} -- gen_loss: {gen_loss:.3f}, dis_loss: {dis_loss:.3f}, gen_acc: {gen_acc:.3f}, dis_acc: {dis_acc:.3f}, it/s: {it_s:.3f}\n"
                logger.info(description)

            if do_validation:
                val_ids = tf.constant(
                    select_batch_ids(arr=wiki_ids, bsz=bsz, seq_len=seq_len)
                )  # [bsz, seq_len]
                val_masked_ids, val_mask = mask_ids(val_ids)
                (
                    val_loss,
                    val_gen_loss,
                    val_dis_loss,
                    val_gen_acc,
                    val_dis_acc,
                    val_gen_ids,
                    val_dis_preds,
                ) = val_step(
                    gen=gen, dis=dis, ids=val_ids, masked_ids=val_masked_ids, mask=val_mask
                )
                log_example(
                    tokenizer, val_ids, val_masked_ids, val_mask, val_gen_ids, val_dis_preds
                )
                description = f"VALIDATION, Step {step} -- val_gen_loss: {val_gen_loss:.3f}, val_dis_loss: {val_dis_loss:.3f}, val_gen_acc: {val_gen_acc:.3f}, val_dis_acc: {val_dis_acc:.3f}\n"
                logger.info(description)

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
