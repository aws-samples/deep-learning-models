"""
albert-base
wrapped_train_step w/ 32bsz, 1acc: 3.13 it/s; 3.51 it/s
wrapped train_step w/ skipping loss/acc allreduce: ? it/s, 3.54 it/s
wrapped train_step w/ skipping xla: ? it/s, 1.50 it/s
wrapped train_batch & wrapped_allreduce: 2.99 it/s, 3.42 it/s

Max per-GPU batch size (albert):
base:
- 512seq:
  - 64 on p3dn w/ 1 grad_acc (? it/s on TF2.1, 1.88 it/s single-node)
  - 32 on p3dn w/ 2 grad_acc (1.72 it/s on TF2.1, 1.82 it/s single-node), 16 on p316
large:
- 512seq: 16 on p3dn w/ 4 grad_acc (0.60 it/s on TF2.1, 0.63 it/s single-node), 8 on p316

Max per-GPU batch size (bert):
base:
- 512seq: 16 on p3dn w/ 4 grad_acc (0.91 it/s on TF 2.1, 1.31 it/s single-node)
large:
- 512seq: 8 on p3dn w/ 8 grad_acc (0.31 it/s on TF 2.1, 0.47 it/s single-node)

A training run of 125k steps is 125k/(1.72 * 3600) ~= 20 hours for base trained on 512seq.
"""

import datetime
import gc
import glob
import logging
import os
import time
from dataclasses import asdict
from typing import Tuple

import numpy as np
import tensorflow as tf
import tqdm
from transformers import (
    AutoConfig,
    GradientAccumulator,
    HfArgumentParser,
    TFAlbertModel,
    TFAutoModelForPreTraining,
    TFBertForPreTraining,
)

from albert.run_squad import get_squad_results_while_pretraining
from common.arguments import (
    DataTrainingArguments,
    LoggingArguments,
    ModelArguments,
    PathArguments,
    TrainingArguments,
)
from common.datasets import get_dataset_from_tfrecords
from common.models import create_model
from common.optimizers import get_adamw_optimizer, get_lamb_optimizer
from common.utils import (
    TqdmLoggingHandler,
    create_tokenizer,
    gather_indexes,
    is_wandb_available,
    rewrap_tf_function,
)

# See https://github.com/huggingface/transformers/issues/3782; this import must come last
import horovod.tensorflow as hvd  # isort:skip

if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)


def mlm_loss_fn(
    prediction_logits: "[batch, max_seq_len (512), vocab_size]",
    label_positions: "[batch, num_masks (20)]",
    label_ids: "[batch, num_masks (20)]",
    label_weights: "[batch, num_masks (20)]",
):
    """ label_weights is either 1 or 0, 0 meaning masked. """
    logits_at_positions = gather_indexes(
        prediction_logits, label_positions
    )  # [b, num_masks, vocab_size]
    preds_at_positions = tf.math.argmax(logits_at_positions, -1)  # [b, num_masks]

    denominator = tf.reduce_sum(label_weights) + 1e-5

    cross_entropy_per_token = label_weights * tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_ids, logits=logits_at_positions
    )  # [b, num_masks]
    cross_entropy_numerator = tf.reduce_sum(cross_entropy_per_token)  # [1]

    accuracy_per_token = label_weights * tf.cast(
        tf.math.equal(label_ids, preds_at_positions), tf.float32
    )  # [b, num_tokens]
    accuracy_numerator = tf.reduce_sum(accuracy_per_token)  # [1]

    cross_entropy = cross_entropy_numerator / denominator
    accuracy = accuracy_numerator / denominator

    return cross_entropy, accuracy


def sop_loss_fn(
    prediction_logits: "[b, 2]",
    next_sentence_labels: "[b, 1]",
):
    """Note that this works for either NSP or SOP, the difference is in how data is generated.
    We want to use SOP objective.
    """
    label_truth = tf.squeeze(next_sentence_labels)  # [b]
    label_preds = tf.math.argmax(prediction_logits, -1)  # [b]

    cross_entropy_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_truth, logits=prediction_logits
    )  # [b]
    accuracy_batch = tf.cast(tf.math.equal(label_truth, label_preds), tf.float32)  # [b]

    cross_entropy = tf.reduce_mean(cross_entropy_batch)  # [1]
    accuracy = tf.reduce_mean(accuracy_batch)  # [b]
    return cross_entropy, accuracy


def train_batch(
    *,
    model,
    optimizer,
    gradient_accumulator,
    input_dict,
    label_positions,
    label_ids,
    label_weights,
    next_sentence_labels,
    skip_sop: bool,
    skip_mlm: bool,
):
    with tf.GradientTape() as tape:
        output = model(input_dict, training=True)
        mlm_logits, sop_logits = output.prediction_logits, output.sop_logits

        # MLM calculation
        if skip_mlm:
            mlm_loss, mlm_acc = tf.constant(0.0), tf.constant(0.0)
        else:
            mlm_loss, mlm_acc = mlm_loss_fn(
                prediction_logits=mlm_logits,
                label_positions=label_positions,
                label_ids=label_ids,
                label_weights=label_weights,
            )
        # SOP calculation
        if skip_sop:
            sop_loss, sop_acc = tf.constant(0.0), tf.constant(0.0)
        else:
            sop_loss, sop_acc = sop_loss_fn(
                prediction_logits=sop_logits, next_sentence_labels=next_sentence_labels
            )
        loss = tf.cast(mlm_loss, dtype=tf.float32) + tf.cast(sop_loss, dtype=tf.float32)
        scaled_loss = optimizer.get_scaled_loss(loss)

    # TODO: On iteration 0, loss=11 and loss_scale()=32768, so scaled_loss=inf.
    # But scaled_grads is not inf, how? tape.gradient() must not be using direct backprop calc
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    gradient_accumulator(scaled_grads)
    return loss, mlm_loss, mlm_acc, sop_loss, sop_acc


def allreduce(model, optimizer, gradient_accumulator, loss, mlm_loss, mlm_acc, sop_loss, sop_acc):
    scaled_grads = gradient_accumulator.gradients
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    # This, which is equivalent to sparse_as_dense=True, gives a mild 2% speedup from 0.62 it/s to 0.63 it/s
    # on BERT-large multinode.
    grads = [
        tf.convert_to_tensor(grad)
        if grad is not None and isinstance(grad, tf.IndexedSlices)
        else grad
        for grad in grads
    ]

    # TODO: Does placing this clip before or after allreduce affect accuracy?
    # Placing before has a regularization effect, no single example can contribute as much.
    # Placing before also gives a 20% speedup when training BERT-large, probably because the
    # gradient operations can be fused by XLA.
    (grads, grad_norm) = tf.clip_by_global_norm(grads, clip_norm=max_grad_norm)

    weight_norm = tf.math.sqrt(
        tf.math.reduce_sum([tf.norm(var, ord=2) ** 2 for var in model.trainable_variables])
    )

    grads = [
        hvd.allreduce(grad, compression=hvd.Compression.fp16) if grad is not None else None
        for grad in grads
    ]

    optimizer.apply_gradients(
        [
            (tf.cast(grad, var.dtype), var)
            for (grad, var) in zip(grads, model.trainable_variables)
            if grad is not None
        ]
    )

    # Clear the gradient accumulator
    gradient_accumulator.reset()

    loss = hvd.allreduce(loss)
    mlm_loss = hvd.allreduce(mlm_loss)
    mlm_acc = hvd.allreduce(mlm_acc)
    sop_loss = hvd.allreduce(sop_loss)
    sop_acc = hvd.allreduce(sop_acc)

    return loss, mlm_loss, mlm_acc, sop_loss, sop_acc, grad_norm, weight_norm


# The bottleneck is here, since each node has 10 GiB/s PCI throughput, accumulating the gradients
# on the CPU. If there's a way to accumulate gradients on the GPU without getting OOM, let's find it!
def train_step(
    model,
    optimizer,
    gradient_accumulator,
    batch,
    gradient_accumulation_steps: int,
    skip_sop: bool,
    skip_mlm: bool,
):
    # MLM uses last_hidden_state, while SOP uses pooler_output
    # Loss is float, acc is half
    total_loss, total_mlm_loss, total_mlm_acc, total_sop_loss, total_sop_acc = (
        tf.constant(0, dtype=tf.float32),
        tf.constant(0, dtype=tf.float32),
        tf.constant(0, dtype=tf.float32),
        tf.constant(0, dtype=tf.float32),
        tf.constant(0, dtype=tf.float32),
    )
    for step in range(gradient_accumulation_steps):
        loss, mlm_loss, mlm_acc, sop_loss, sop_acc = train_batch(
            model=model,
            optimizer=optimizer,
            gradient_accumulator=gradient_accumulator,
            input_dict={
                "input_ids": batch["input_ids"][step],
                "attention_mask": batch["input_mask"][step],
                "token_type_ids": batch["segment_ids"][step],
            },
            label_positions=batch["masked_lm_positions"][step],
            label_ids=batch["masked_lm_ids"][step],
            label_weights=batch["masked_lm_weights"][step],
            next_sentence_labels=batch["next_sentence_labels"][step],
            skip_sop=skip_sop,
            skip_mlm=skip_mlm,
        )
        total_loss += tf.cast(loss, total_loss.dtype)
        total_mlm_loss += tf.cast(mlm_loss, total_mlm_loss.dtype)
        total_mlm_acc += tf.cast(mlm_acc, total_mlm_acc.dtype)
        total_sop_loss += tf.cast(sop_loss, total_sop_loss.dtype)
        total_sop_acc += tf.cast(sop_acc, total_sop_acc.dtype)

    total_loss /= gradient_accumulation_steps
    total_mlm_loss /= gradient_accumulation_steps
    total_mlm_acc /= gradient_accumulation_steps
    total_sop_loss /= gradient_accumulation_steps
    total_sop_acc /= gradient_accumulation_steps

    return_tuple = allreduce(
        model=model,
        optimizer=optimizer,
        gradient_accumulator=gradient_accumulator,
        loss=total_loss,
        mlm_loss=total_mlm_loss,
        mlm_acc=total_mlm_acc,
        sop_loss=total_sop_loss,
        sop_acc=total_sop_acc,
    )
    return return_tuple


def validation_batch(model, batch, skip_mlm: bool, skip_sop: bool):
    input_ids = batch["input_ids"]
    attention_mask = batch["input_mask"]
    token_type_ids = batch["segment_ids"]
    label_positions = batch["masked_lm_positions"]
    label_ids = batch["masked_lm_ids"]
    label_weights = batch["masked_lm_weights"]
    next_sentence_labels = batch["next_sentence_labels"]

    input_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    output = model(input_dict, training=False)  # ([b,seq,vocab_size], [b,2])
    mlm_logits, sop_logits = output.prediction_logits, output.sop_logits

    # MLM calculation
    if skip_mlm:
        mlm_loss, mlm_acc = 0, 0
    else:
        mlm_loss, mlm_acc = mlm_loss_fn(
            prediction_logits=mlm_logits,
            label_positions=label_positions,
            label_ids=label_ids,
            label_weights=label_weights,
        )
    # SOP calculation
    if skip_sop:
        sop_loss, sop_acc = 0, 0
    else:
        sop_loss, sop_acc = sop_loss_fn(
            prediction_logits=sop_logits, next_sentence_labels=next_sentence_labels
        )
    loss = mlm_loss + sop_loss  # Should there be a coefficient on one of these?
    return loss, mlm_loss, mlm_acc, sop_loss, sop_acc


def run_validation(model, validation_dataset, skip_sop: bool, skip_mlm: bool):
    num_batches = 100
    val_loss, val_mlm_loss, val_mlm_acc, val_sop_loss, val_sop_acc = (0, 0, 0, 0, 0)
    for batch in validation_dataset.take(num_batches):
        loss, mlm_loss, mlm_acc, sop_loss, sop_acc = validation_batch(
            model=model, batch=batch, skip_sop=skip_sop, skip_mlm=skip_mlm
        )
        val_loss += loss
        val_mlm_loss += mlm_loss
        val_mlm_acc += mlm_acc
        val_sop_loss += sop_loss
        val_sop_acc += sop_acc

    val_loss /= num_batches
    val_mlm_loss /= num_batches
    val_mlm_acc /= num_batches
    val_sop_loss /= num_batches
    val_sop_acc /= num_batches

    return (val_loss, val_mlm_loss, val_mlm_acc, val_sop_loss, val_sop_acc)


def wrap_global_functions(do_gradient_accumulation: bool):
    global validation_batch
    validation_batch = rewrap_tf_function(validation_batch)
    if do_gradient_accumulation:
        global train_batch
        train_batch = rewrap_tf_function(train_batch)
        global allreduce
        allreduce = rewrap_tf_function(allreduce)
    else:
        global train_step
        train_step = rewrap_tf_function(train_step)


def get_checkpoint_paths_from_prefix(prefix: str) -> Tuple[str, str]:
    """ Returns the model_ckpt path and optimizer_ckpt path. """
    return f"{prefix}.ckpt", f"{prefix}-optimizer.npy"


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

    tf.random.set_seed(train_args.seed)
    tf.autograph.set_verbosity(0)

    # Settings init
    parse_bool = lambda arg: arg == "true"
    do_gradient_accumulation = train_args.gradient_accumulation_steps > 1
    do_xla = not parse_bool(train_args.skip_xla)
    do_eager = parse_bool(train_args.eager)
    skip_sop = parse_bool(train_args.skip_sop)
    skip_mlm = parse_bool(train_args.skip_mlm)
    pre_layer_norm = parse_bool(model_args.pre_layer_norm)
    fast_squad = parse_bool(log_args.fast_squad)
    dummy_eval = parse_bool(log_args.dummy_eval)
    is_sagemaker = path_args.filesystem_prefix.startswith("/opt/ml")
    disable_tqdm = is_sagemaker
    global max_grad_norm
    max_grad_norm = train_args.max_grad_norm

    # Horovod init
    hvd.init()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    # XLA, AutoGraph
    tf.config.optimizer.set_jit(do_xla)
    tf.config.experimental_run_functions_eagerly(do_eager)

    if hvd.rank() == 0:
        # Run name should only be used on one process to avoid race conditions
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        platform = "sm" if is_sagemaker else "eks"
        if skip_sop:
            loss_str = "-skipsop"
        elif skip_mlm:
            loss_str = "-skipmlm"
        else:
            loss_str = ""

        if log_args.run_name is None:
            metadata = (
                f"{model_args.model_type}"
                f"-{model_args.model_size}"
                f"-{model_args.load_from}"
                f"-{hvd.size()}gpus"
                f"-{train_args.per_gpu_batch_size * hvd.size() * train_args.gradient_accumulation_steps}globalbatch"
                f"-{train_args.learning_rate}maxlr"
                f"-{train_args.learning_rate_decay_power}power"
                f"-{train_args.optimizer}opt"
                f"-{train_args.total_steps}steps"
                f"-{'preln' if pre_layer_norm else 'postln'}"
                f"{loss_str}"
                f"-{model_args.hidden_dropout_prob}dropout"
            )
            run_name = f"{current_time}-{platform}-{metadata}-{train_args.name if train_args.name else 'unnamed'}"
        else:
            run_name = log_args.run_name

        # Logging should only happen on a single process
        # https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
        level = logging.INFO
        format = "%(asctime)-15s %(name)-12s: %(levelname)-8s %(message)s"
        handlers = [
            logging.FileHandler(
                os.path.join(path_args.filesystem_prefix, path_args.log_dir, f"{run_name}.log")
            ),
            TqdmLoggingHandler(),
        ]
        logging.basicConfig(level=level, format=format, handlers=handlers)

        # Check that arguments passed in properly, only after registering the alert_func and logging
        assert not (skip_sop and skip_mlm), "Cannot use --skip_sop and --skip_mlm"

    wrap_global_functions(do_gradient_accumulation)

    # Create optimizer and enable AMP loss scaling.
    if train_args.optimizer == "lamb":
        optimizer = get_lamb_optimizer(train_args)
    elif train_args.optimizer == "adamw":
        optimizer = get_adamw_optimizer(train_args)

    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
        optimizer, loss_scale="dynamic"
    )
    gradient_accumulator = GradientAccumulator()

    loaded_optimizer_weights = None

    model = create_model(model_class=TFAutoModelForPreTraining, model_args=model_args)
    tokenizer = create_tokenizer(model_args.model_type)
    if model_args.load_from == "checkpoint":
        checkpoint_path = os.path.join(path_args.filesystem_prefix, model_args.checkpoint_path)
        model_ckpt, optimizer_ckpt = get_checkpoint_paths_from_prefix(checkpoint_path)
        if hvd.rank() == 0:
            model.load_weights(model_ckpt)
            if model_args.load_optimizer_state == "true":
                loaded_optimizer_weights = np.load(optimizer_ckpt, allow_pickle=True)
            # We do not set the weights yet, we have to do a first step to initialize the optimizer.

    # Train filenames are [1, 2047], Val filenames are [0]. Note the different subdirectories
    # Move to same folder structure and remove if/else
    train_glob = os.path.join(path_args.filesystem_prefix, path_args.train_dir, "*.tfrecord")
    validation_glob = os.path.join(path_args.filesystem_prefix, path_args.val_dir, "*.tfrecord")

    train_filenames = glob.glob(train_glob)
    validation_filenames = glob.glob(validation_glob)

    train_dataset = get_dataset_from_tfrecords(
        model_type=model_args.model_type,
        filenames=train_filenames,
        max_seq_length=data_args.max_seq_length,
        max_predictions_per_seq=data_args.max_predictions_per_seq,
        per_gpu_batch_size=train_args.per_gpu_batch_size,
    )  # Of shape [per_gpu_batch_size, ...]
    # Batch of batches, helpful for gradient accumulation. Shape [grad_steps, per_gpu_batch_size, ...]
    train_dataset = train_dataset.batch(train_args.gradient_accumulation_steps)
    # One iteration with 10 dupes, 8 nodes seems to be 60-70k steps.
    train_dataset = train_dataset.prefetch(buffer_size=8)

    # Validation should only be done on one node, since Horovod doesn't allow allreduce on a subset of ranks
    if hvd.rank() == 0:
        validation_dataset = get_dataset_from_tfrecords(
            model_type=model_args.model_type,
            filenames=validation_filenames,
            max_seq_length=data_args.max_seq_length,
            max_predictions_per_seq=data_args.max_predictions_per_seq,
            per_gpu_batch_size=train_args.per_gpu_batch_size,
        )
        # validation_dataset = validation_dataset.batch(1)
        validation_dataset = validation_dataset.prefetch(buffer_size=8)

        pbar = tqdm.tqdm(total=train_args.total_steps, disable=disable_tqdm)
        summary_writer = None  # Only create a writer if we make it through a successful step
        logger.info(f"Starting training, job name {run_name}")

    i = 1
    start_time = time.perf_counter()
    for batch in train_dataset:
        learning_rate = optimizer.learning_rate(step=tf.constant(i, dtype=tf.float32))
        # weight_decay = wd_schedule(step=tf.constant(i, dtype=tf.float32))
        loss_scale = optimizer.loss_scale
        loss, mlm_loss, mlm_acc, sop_loss, sop_acc, grad_norm, weight_norm = train_step(
            model=model,
            optimizer=optimizer,
            gradient_accumulator=gradient_accumulator,
            batch=batch,
            gradient_accumulation_steps=train_args.gradient_accumulation_steps,
            skip_sop=skip_sop,
            skip_mlm=skip_mlm,
        )

        # Don't want to wrap broadcast_variables() in a tf.function, can lead to asynchronous errors
        if i == 1:
            if hvd.rank() == 0 and loaded_optimizer_weights is not None:
                optimizer.set_weights(loaded_optimizer_weights)
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)
            i = optimizer.get_weights()[0]

        is_final_step = i >= train_args.total_steps
        do_squad = (log_args.squad_frequency != 0) and (
            (i % log_args.squad_frequency == 0) or is_final_step
        )
        # Squad requires all the ranks to train, but results are only returned on rank 0
        if do_squad:
            squad_results = get_squad_results_while_pretraining(
                model=model,
                tokenizer=tokenizer,
                model_size=model_args.model_size,
                filesystem_prefix=path_args.filesystem_prefix,
                step=i,
                dataset=data_args.squad_version,
                fast=log_args.fast_squad,
                dummy_eval=log_args.dummy_eval,
            )
            if hvd.rank() == 0:
                squad_exact, squad_f1 = squad_results["exact"], squad_results["f1"]
                logger.info(f"SQuAD step {i} -- F1: {squad_f1:.3f}, Exact: {squad_exact:.3f}")
            # Re-wrap autograph so it doesn't get arg mismatches
            wrap_global_functions(do_gradient_accumulation)
            gc.collect()

        if hvd.rank() == 0:
            do_log = i % log_args.log_frequency == 0
            do_checkpoint = (log_args.checkpoint_frequency != 0) and (
                (i % log_args.checkpoint_frequency == 0) or is_final_step
            )
            do_validation = (log_args.validation_frequency != 0) and (
                (i % log_args.validation_frequency == 0) or is_final_step
            )

            pbar.update(1)
            description = f"Loss: {loss:.3f}, MLM: {mlm_loss:.3f}, SOP: {sop_loss:.3f}, MLM_acc: {mlm_acc:.3f}, SOP_acc: {sop_acc:.3f}"
            pbar.set_description(description)
            if do_log:
                elapsed_time = time.perf_counter() - start_time
                if i == 1:
                    logger.info(f"First step: {elapsed_time:.3f} secs")
                else:
                    it_per_sec = log_args.log_frequency / elapsed_time
                    logger.info(f"Train step {i} -- {description} -- It/s: {it_per_sec:.2f}")
                    start_time = time.perf_counter()

            if do_checkpoint:
                checkpoint_prefix = os.path.join(
                    path_args.filesystem_prefix, path_args.checkpoint_dir, f"{run_name}-step{i}"
                )
                model_ckpt = f"{checkpoint_prefix}.ckpt"
                optimizer_ckpt = f"{checkpoint_prefix}-optimizer.npy"
                logger.info(f"Saving model at {model_ckpt}, optimizer at {optimizer_ckpt}")
                model.save_weights(model_ckpt)
                # model.load_weights(model_ckpt)

                optimizer_weights = optimizer.get_weights()
                np.save(optimizer_ckpt, optimizer_weights)
                # optimizer.set_weights(optimizer_weights)

            if do_validation:
                val_loss, val_mlm_loss, val_mlm_acc, val_sop_loss, val_sop_acc = run_validation(
                    model=model,
                    validation_dataset=validation_dataset,
                    skip_sop=skip_sop,
                    skip_mlm=skip_mlm,
                )
                description = f"Loss: {val_loss:.3f}, MLM: {val_mlm_loss:.3f}, SOP: {val_sop_loss:.3f}, MLM_acc: {val_mlm_acc:.3f}, SOP_acc: {val_sop_acc:.3f}"
                logger.info(f"Validation step {i} -- {description}")

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
                    "global_batch_size": train_args.per_gpu_batch_size * hvd.size(),
                }
                if is_wandb_available():
                    wandb.init(config=config, project=model_args.model_type)
                    wandb.run.save()
                    wandb_run_name = wandb.run.name

            train_metrics = {
                "weight_norm": weight_norm,
                "grad_norm": grad_norm,
                "loss_scale": loss_scale,
                "learning_rate": learning_rate,
                "train/loss": loss,
                "train/mlm_loss": mlm_loss,
                "train/mlm_acc": mlm_acc,
                "train/sop_loss": sop_loss,
                "train/sop_acc": sop_acc,
            }
            all_metrics = {**train_metrics}
            if do_validation:
                val_metrics = {
                    "val/loss": val_loss,
                    "val/mlm_loss": val_mlm_loss,
                    "val/mlm_acc": val_mlm_acc,
                    "val/sop_loss": val_sop_loss,
                    "val/sop_acc": val_sop_acc,
                }
                all_metrics = {**all_metrics, **val_metrics}
            if do_squad:
                squad_metrics = {
                    "squad/f1": squad_f1,
                    "squad/exact": squad_exact,
                }
                all_metrics = {**all_metrics, **squad_metrics}

            # Log to TensorBoard
            with summary_writer.as_default():
                for name, val in all_metrics.items():
                    tf.summary.scalar(name, val, step=i)
            # Log to Weights & Biases
            if is_wandb_available():
                wandb.log({"step": i, **all_metrics})

        i += 1
        if is_final_step:
            break

    if hvd.rank() == 0:
        pbar.close()
        logger.info(f"Finished pretraining, job name {run_name}")


if __name__ == "__main__":
    main()
