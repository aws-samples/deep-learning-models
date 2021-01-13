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
import glob
import logging
import os
import time
from collections import namedtuple
from dataclasses import asdict
from typing import Tuple

import numpy as np
import tensorflow as tf
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
    PathArguments,
    TrainingArguments,
)
from common.datasets import get_dataset_from_tfrecords
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


def generate_corruption_mask(ids, attention_mask):
    mask = (
        tf.cast(tf.random.uniform(shape=ids.shape) > 0.85, dtype=attention_mask.dtype)
        * attention_mask
    )
    return mask


def mask_ids(ids, corruption_mask, mask_id):
    return tf.where(tf.cast(corruption_mask, tf.bool), tf.cast(mask_id, dtype=ids.dtype), ids)


# TODO: Make temperature a hyperparameter
def temperature_sampling(logits, output_type, per_gpu_batch_size, max_seq_length, temperature):
    if temperature is None or temperature == 0.0:
        return tf.argmax(logits, axis=-1, output_type=output_type)

    logger.info(f"temperature {temperature}")

    log_prob = tf.nn.log_softmax(logits / temperature)
    log_prob = tf.reshape(
        log_prob, [per_gpu_batch_size * max_seq_length, -1], name=None
    )  # [bsz, seq_len]
    preds = tf.cast(tf.random.categorical(log_prob, 1), tf.int64)
    preds = tf.reshape(preds, [per_gpu_batch_size, max_seq_length], name=None)  # [bsz, seq_len]
    return preds


def forward(
    gen,
    dis,
    ids,
    masked_ids,
    attention_mask,
    corruption_mask,
    per_gpu_batch_size,
    max_seq_length,
    return_preds=False,
):
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
    # adv_ids = tf.argmax(gen_logits, axis=-1, output_type=ids.dtype)  # [bsz, seq_len]
    adv_ids = temperature_sampling(
        logits=gen_logits,
        output_type=ids.dtype,
        per_gpu_batch_size=per_gpu_batch_size,
        max_seq_length=max_seq_length,
        temperature=1.0,
    )
    ids_equal = tf.cast(adv_ids == ids, dtype=tf.int64)  # [bsz, seq_len]
    gen_correct = tf.boolean_mask(ids_equal, corruption_mask)  # [bsz * n_masks]
    gen_acc = tf.reduce_mean(tf.cast(gen_correct, dtype=tf.float32))  # [1]

    # Discriminator loss
    gen_ids = corruption_mask * adv_ids + (1 - corruption_mask) * ids  # [bsz, seq_len]
    (dis_logits,) = dis({"input_ids": gen_ids, "attention_mask": attention_mask})  # [bsz, seq_len]
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

    res = (loss, gen_loss, dis_loss, gen_acc, dis_acc)
    if return_preds:
        res += (gen_ids, dis_preds)
    return res


@tf.function
def val_step(gen, dis, ids, attention_mask, per_gpu_batch_size, max_seq_length, mask_token_id: int):
    corruption_mask = generate_corruption_mask(ids=ids, attention_mask=attention_mask)
    masked_ids = mask_ids(ids=ids, corruption_mask=corruption_mask, mask_id=mask_token_id)
    loss, gen_loss, dis_loss, gen_acc, dis_acc, gen_ids, dis_preds = forward(
        gen=gen,
        dis=dis,
        ids=ids,
        masked_ids=masked_ids,
        attention_mask=attention_mask,
        corruption_mask=corruption_mask,
        return_preds=True,
    )
    Output = namedtuple(
        "Output",
        [
            "loss",
            "gen_loss",
            "dis_loss",
            "gen_acc",
            "dis_acc",
            "corruption_mask",
            "masked_ids",
            "gen_ids",
            "dis_preds",
        ],
    )
    return Output(
        loss=loss,
        gen_loss=gen_loss,
        dis_loss=dis_loss,
        gen_acc=gen_acc,
        dis_acc=dis_acc,
        corruption_mask=corruption_mask,
        masked_ids=masked_ids,
        gen_ids=gen_ids,
        dis_preds=dis_preds,
    )


@tf.function
def train_step(
    optimizer, gen, dis, ids, attention_mask, per_gpu_batch_size, max_seq_length, mask_token_id: int
):
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
    corruption_mask = generate_corruption_mask(ids=ids, attention_mask=attention_mask)
    masked_ids = mask_ids(ids=ids, corruption_mask=corruption_mask, mask_id=mask_token_id)
    with tf.GradientTape() as tape:
        loss, gen_loss, dis_loss, gen_acc, dis_acc = forward(
            gen=gen,
            dis=dis,
            ids=ids,
            masked_ids=masked_ids,
            attention_mask=attention_mask,
            per_gpu_batch_size=per_gpu_batch_size,
            max_seq_length=max_seq_length,
            corruption_mask=corruption_mask,
        )

    # Be careful not to double-apply gradients by having duplicate embeddings in the variable list
    # See https://github.com/tensorflow/tensorflow/issues/30712
    vars = gen.trainable_variables + dis.trainable_variables
    # Tensor is unhashable, so can't do list(set(vars))
    # Relies on all ranks doing the same list->dict->list ordering
    # Ensure that this happens every time, not just during function tracing
    # tf.print(f"vars has length {len(vars)} before dedupe")
    vars = list({var.experimental_ref(): var for var in vars}.values())
    # tf.print(f"vars has length {len(vars)} after dedupe")
    grads = tape.gradient(loss, vars)
    # This sparse_as_dense speedup is absolutely necessary here, even though it doesn't make a difference for ALBERT
    grads = [
        tf.convert_to_tensor(grad)
        if grad is not None and isinstance(grad, tf.IndexedSlices)
        else grad
        for grad in grads
    ]
    grads = [
        hvd.allreduce(grad, compression=hvd.Compression.fp16) if grad is not None else None
        for grad in grads
    ]
    optimizer.apply_gradients(zip(grads, vars))

    Output = namedtuple("Output", ["loss", "gen_loss", "dis_loss", "gen_acc", "dis_acc"])
    return Output(loss=loss, gen_loss=gen_loss, dis_loss=dis_loss, gen_acc=gen_acc, dis_acc=dis_acc)


def get_checkpoint_paths_from_prefix(prefix: str) -> Tuple[str, str, str]:
    """ Returns the model_ckpt path and optimizer_ckpt path. """
    return f"{prefix}-discriminator.ckpt", f"{prefix}-generator.ckpt", f"{prefix}-optimizer.npy"


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments, PathArguments)
    )
    (
        model_args,
        data_args,
        train_args,
        log_args,
        path_args,
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

    gen_config = ElectraConfig.from_pretrained(f"google/electra-{model_args.model_size}-generator")
    dis_config = ElectraConfig.from_pretrained(
        f"google/electra-{model_args.model_size}-discriminator"
    )

    gen = TFElectraForMaskedLM(config=gen_config)
    dis = TFElectraForPreTraining(config=dis_config)
    optimizer = get_adamw_optimizer(train_args)

    # Tie the weights
    if model_args.electra_tie_weights == "true":
        gen.electra.embeddings = dis.electra.embeddings

    loaded_optimizer_weights = None
    if model_args.load_from == "checkpoint":
        checkpoint_path = os.path.join(path_args.filesystem_prefix, model_args.checkpoint_path)
        dis_ckpt, gen_ckpt, optimizer_ckpt = get_checkpoint_paths_from_prefix(checkpoint_path)
        if hvd.rank() == 0:
            dis.load_weights(dis_ckpt)
            gen.load_weights(gen_ckpt)
            loaded_optimizer_weights = np.load(optimizer_ckpt, allow_pickle=True)

    start_time = time.perf_counter()

    if hvd.rank() == 0:
        # Logging should only happen on a single process
        # https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
        level = logging.INFO
        format = "%(asctime)-15s %(name)-12s: %(levelname)-8s %(message)s"
        handlers = [
            TqdmLoggingHandler(),
        ]
        summary_writer = None  # Only create a writer if we make it through a successful step
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
            run_name = log_args.run_name

    logger.info(f"Training with dataset at {path_args.train_dir}")
    logger.info(f"Validating with dataset at {path_args.val_dir}")

    train_glob = os.path.join(path_args.filesystem_prefix, path_args.train_dir, "*.tfrecord*")
    validation_glob = os.path.join(path_args.filesystem_prefix, path_args.val_dir, "*.tfrecord*")

    train_filenames = glob.glob(train_glob)
    validation_filenames = glob.glob(validation_glob)
    logger.info(
        f"Number of train files {len(train_filenames)}, number of validation files {len(validation_filenames)}"
    )

    tf_train_dataset = get_dataset_from_tfrecords(
        model_type=model_args.model_type,
        filenames=train_filenames,
        per_gpu_batch_size=train_args.per_gpu_batch_size,
        max_seq_length=data_args.max_seq_length,
    )

    tf_train_dataset = tf_train_dataset.prefetch(buffer_size=8)

    if hvd.rank() == 0:
        tf_val_dataset = get_dataset_from_tfrecords(
            model_type=model_args.model_type,
            filenames=validation_filenames,
            per_gpu_batch_size=train_args.per_gpu_batch_size,
            max_seq_length=data_args.max_seq_length,
        )
        tf_val_dataset = tf_val_dataset.prefetch(buffer_size=8)

    wandb_run_name = None

    step = 1
    for batch in tf_train_dataset:
        learning_rate = optimizer.learning_rate(step=tf.constant(step, dtype=tf.float32))
        ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        train_result = train_step(
            optimizer=optimizer,
            gen=gen,
            dis=dis,
            ids=ids,
            attention_mask=attention_mask,
            per_gpu_batch_size=train_args.per_gpu_batch_size,
            max_seq_length=data_args.max_seq_length,
            mask_token_id=tokenizer.mask_token_id,
        )

        if step == 1:
            # Horovod broadcast
            if hvd.rank() == 0 and loaded_optimizer_weights is not None:
                optimizer.set_weights(loaded_optimizer_weights)
            hvd.broadcast_variables(gen.variables, root_rank=0)
            hvd.broadcast_variables(dis.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)
            step = optimizer.get_weights()[0]

        is_final_step = step >= train_args.total_steps
        if hvd.rank() == 0:
            do_log = step % log_args.log_frequency == 0
            do_checkpoint = (step > 1) and (
                (step % log_args.checkpoint_frequency == 0) or is_final_step
            )
            do_validation = step % log_args.validation_frequency == 0

            if do_log:
                elapsed_time = time.perf_counter() - start_time  # Off for first log
                it_s = log_args.log_frequency / elapsed_time
                start_time = time.perf_counter()
                description = f"Step {step} -- gen_loss: {train_result.gen_loss:.3f}, dis_loss: {train_result.dis_loss:.3f}, gen_acc: {train_result.gen_acc:.3f}, dis_acc: {train_result.dis_acc:.3f}, it/s: {it_s:.3f}\n"
                logger.info(description)

            if do_validation:
                for batch in tf_val_dataset.take(1):
                    val_ids = batch["input_ids"]
                    val_attention_mask = batch["attention_mask"]
                    val_result = val_step(
                        gen=gen,
                        dis=dis,
                        ids=val_ids,
                        attention_mask=val_attention_mask,
                        per_gpu_batch_size=train_args.per_gpu_batch_size,
                        max_seq_length=data_args.max_seq_length,
                        mask_token_id=tokenizer.mask_token_id,
                    )
                    log_example(
                        tokenizer,
                        val_ids,
                        val_result.masked_ids,
                        val_result.corruption_mask,
                        val_result.gen_ids,
                        val_result.dis_preds,
                    )
                    description = f"VALIDATION, Step {step} -- val_gen_loss: {val_result.gen_loss:.3f}, val_dis_loss: {val_result.dis_loss:.3f}, val_gen_acc: {val_result.gen_acc:.3f}, val_dis_acc: {val_result.dis_acc:.3f}\n"
                    logger.info(description)

            train_metrics = {
                "learning_rate": learning_rate,
                "train/loss": train_result.loss,
                "train/gen_loss": train_result.gen_loss,
                "train/dis_loss": train_result.dis_loss,
                "train/gen_acc": train_result.gen_acc,
                "train/dis_acc": train_result.dis_acc,
            }
            all_metrics = {**train_metrics}
            if do_validation:
                val_metrics = {
                    "val/loss": val_result.loss,
                    "val/gen_loss": val_result.gen_loss,
                    "val/dis_loss": val_result.dis_loss,
                    "val/gen_acc": val_result.gen_acc,
                    "val/dis_acc": val_result.dis_acc,
                }
                all_metrics = {**all_metrics, **val_metrics}
            if do_log:
                all_metrics = {"it_s": it_s, **all_metrics}

            if is_wandb_available():
                if wandb_run_name is None:
                    config = {
                        **asdict(model_args),
                        **asdict(data_args),
                        **asdict(train_args),
                        **asdict(log_args),
                        **asdict(path_args),
                        "global_batch_size": train_args.per_gpu_batch_size * hvd.size(),
                        "n_gpus": hvd.size(),
                    }
                    wandb.init(config=config, project="electra")
                    wandb.run.save()
                    wandb_run_name = wandb.run.name
                wandb.log({"step": step, **all_metrics})

                # Create summary_writer after the first step
            if summary_writer is None:
                summary_writer = tf.summary.create_file_writer(
                    os.path.join(path_args.filesystem_prefix, path_args.log_dir, run_name)
                )
                config = {
                    **asdict(model_args),
                    **asdict(data_args),
                    **asdict(train_args),
                    **asdict(log_args),
                    **asdict(path_args),
                    "global_batch_size": train_args.per_gpu_batch_size * hvd.size(),
                    "n_gpus": hvd.size(),
                }

            # Log to TensorBoard
            with summary_writer.as_default():
                for name, val in all_metrics.items():
                    tf.summary.scalar(name, val, step=step)

            if do_checkpoint:
                dis_model_ckpt = os.path.join(
                    path_args.filesystem_prefix,
                    path_args.checkpoint_dir,
                    f"{run_name}-step{step}-discriminator.ckpt",
                )
                gen_model_ckpt = os.path.join(
                    path_args.filesystem_prefix,
                    path_args.checkpoint_dir,
                    f"{run_name}-step{step}-generator.ckpt",
                )
                optimizer_ckpt = os.path.join(
                    path_args.filesystem_prefix,
                    path_args.checkpoint_dir,
                    f"{run_name}-step{step}-optimizer.npy",
                )
                logger.info(
                    f"Saving discriminator model at {dis_model_ckpt}, generator model at {gen_model_ckpt}, optimizer at {optimizer_ckpt}"
                )
                dis.save_weights(dis_model_ckpt)
                gen.save_weights(gen_model_ckpt)
                np.save(optimizer_ckpt, optimizer.get_weights())

        step += 1
        if is_final_step:
            break


if __name__ == "__main__":
    main()
