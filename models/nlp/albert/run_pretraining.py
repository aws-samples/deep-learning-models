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


import argparse
import datetime
import glob
import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tqdm
from tensorflow_addons.optimizers import LAMB, AdamW
from transformers import (
    AutoConfig,
    GradientAccumulator,
    TFAlbertModel,
    TFAutoModelForPreTraining,
    TFBertForPreTraining,
)

from datasets import get_mlm_dataset
from learning_rate_schedules import LinearWarmupPolyDecaySchedule
from run_squad import get_squad_results_while_pretraining
from utils import gather_indexes, rewrap_tf_function

# See https://github.com/huggingface/transformers/issues/3782; this import must come last
import horovod.tensorflow as hvd  # isort:skip

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
    prediction_logits: "[b, 2]", next_sentence_labels: "[b, 1]",
):
    """ Note that this works for either NSP or SOP, the difference is in how data is generated.
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
    opt,
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
        mlm_logits, sop_logits = model(input_dict, training=True)

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
        scaled_loss = opt.get_scaled_loss(loss)

    # TODO: On iteration 0, loss=11 and loss_scale()=32768, so scaled_loss=inf.
    # But scaled_grads is not inf, how? tape.gradient() must not be using direct backprop calc
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    gradient_accumulator(scaled_grads)
    return loss, mlm_loss, mlm_acc, sop_loss, sop_acc


def allreduce(model, opt, gradient_accumulator, loss, mlm_loss, mlm_acc, sop_loss, sop_acc):
    scaled_grads = gradient_accumulator.gradients
    grads = opt.get_unscaled_gradients(scaled_grads)
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
    (grads, grad_norm) = tf.clip_by_global_norm(grads, clip_norm=args.max_grad_norm)
    weight_norm = tf.math.sqrt(
        tf.math.reduce_sum([tf.norm(var, ord=2) ** 2 for var in model.trainable_variables])
    )

    grads = [
        hvd.allreduce(grad, compression=hvd.Compression.fp16) if grad is not None else None
        for grad in grads
    ]

    opt.apply_gradients(
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
    opt,
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
            opt=opt,
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
        opt=opt,
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
    mlm_logits, sop_logits = model(input_dict, training=False)  # ([b,seq,vocab_size], [b,2])

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
    # A single TFRecord shard contains 22663 batches, or 170k examples.
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
    """ Returns the model_ckpt path and opt_ckpt path. """
    return f"{prefix}.ckpt", f"{prefix}-opt.npy"


def main(
    fsx_prefix: str,
    load_from: str,
    checkpoint_path: str,
    model_type: str,
    model_size: str,
    batch_size: int,
    max_seq_length: int,
    gradient_accumulation_steps: int,
    optimizer: str,
    name: str,
    learning_rate: float,
    end_learning_rate: float,
    learning_rate_decay_power: float,
    warmup_steps: int,
    total_steps: int,
    skip_sop: bool,
    skip_mlm: bool,
    pre_layer_norm: bool,
    fast_squad: bool,
    dummy_eval: bool,
    squad_steps: List[int],
    hidden_dropout_prob: float,
    seed: int,
):
    # Hard-coded values that don't need to be arguments
    max_predictions_per_seq = 20
    log_frequency = 1000
    checkpoint_frequency = 5000
    validate_frequency = 2000
    histogram_frequency = 100
    do_gradient_accumulation = gradient_accumulation_steps > 1

    if hvd.rank() == 0:
        # Run name should only be used on one process to avoid race conditions
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        platform = "eks" if args.fsx_prefix == "/fsx" else "sm"
        if skip_sop:
            loss_str = "-skipsop"
        elif skip_mlm:
            loss_str = "-skipmlm"
        else:
            loss_str = ""

        metadata = (
            f"{model_type}"
            f"-{model_size}"
            f"-{load_from}"
            f"-{hvd.size()}gpus"
            f"-{batch_size}batch"
            f"-{gradient_accumulation_steps}accum"
            f"-{learning_rate}maxlr"
            f"-{end_learning_rate}endlr"
            f"-{learning_rate_decay_power}power"
            f"-{args.max_grad_norm}maxgrad"
            f"-{optimizer}opt"
            f"-{total_steps}steps"
            f"-{max_seq_length}seq"
            f"-{'preln' if pre_layer_norm else 'postln'}"
            f"{loss_str}"
            f"-{hidden_dropout_prob}dropout"
            f"-{seed}seed"
        )
        run_name = f"{current_time}-{platform}-{metadata}-{name if name else 'unnamed'}"

        # Logging should only happen on a single process
        # https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
        level = logging.INFO
        format = "%(asctime)-15s %(name)-12s: %(levelname)-8s %(message)s"
        handlers = [
            logging.FileHandler(f"{fsx_prefix}/logs/albert/{run_name}.log"),
            logging.StreamHandler(),
        ]
        logging.basicConfig(level=level, format=format, handlers=handlers)

        # Check that arguments passed in properly, only after registering the alert_func and logging
        assert not (skip_sop and skip_mlm), "Cannot use --skip_sop and --skip_mlm"

    wrap_global_functions(do_gradient_accumulation)

    if model_type == "albert":
        model_desc = f"albert-{model_size}-v2"
    elif model_type == "bert":
        model_desc = f"bert-{model_size}-uncased"

    config = AutoConfig.from_pretrained(model_desc)
    config.pre_layer_norm = pre_layer_norm
    config.hidden_dropout_prob = hidden_dropout_prob
    model = TFAutoModelForPreTraining.from_config(config)

    # Create optimizer and enable AMP loss scaling.
    schedule = LinearWarmupPolyDecaySchedule(
        max_learning_rate=learning_rate,
        end_learning_rate=end_learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        power=learning_rate_decay_power,
    )
    if optimizer == "lamb":
        opt = LAMB(
            learning_rate=schedule,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )
    elif optimizer == "adam":
        opt = AdamW(weight_decay=0.0, learning_rate=schedule)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale="dynamic")
    gradient_accumulator = GradientAccumulator()

    if load_from == "scratch":
        pass
    elif load_from.startswith("huggingface"):
        assert model_type == "albert", "Only loading pretrained albert models is supported"
        huggingface_name = f"albert-{model_size}-v2"
        if load_from == "huggingface":
            albert = TFAlbertModel.from_pretrained(huggingface_name, config=config)
            model.albert = albert
    else:
        model_ckpt, opt_ckpt = get_checkpoint_paths_from_prefix(checkpoint_path)

        model = TFAutoModelForPreTraining.from_config(config)
        if hvd.rank() == 0:
            model.load_weights(model_ckpt)
            loaded_opt_weights = np.load(opt_ckpt, allow_pickle=True)
            # We do not set the weights yet, we have to do a first step to initialize the optimizer.

    # Train filenames are [1, 2047], Val filenames are [0]. Note the different subdirectories
    train_glob = f"{fsx_prefix}/albert_pretraining/tfrecords/train/max_seq_len_{max_seq_length}_max_predictions_per_seq_{max_predictions_per_seq}_masked_lm_prob_15/albert_*.tfrecord"
    validation_glob = f"{fsx_prefix}/albert_pretraining/tfrecords/validation/max_seq_len_{max_seq_length}_max_predictions_per_seq_{max_predictions_per_seq}_masked_lm_prob_15/albert_*.tfrecord"

    train_filenames = glob.glob(train_glob)
    validation_filenames = glob.glob(validation_glob)

    train_dataset = get_mlm_dataset(
        filenames=train_filenames,
        max_seq_length=max_seq_length,
        max_predictions_per_seq=max_predictions_per_seq,
        batch_size=batch_size,
    )  # Of shape [batch_size, ...]
    train_dataset = train_dataset.batch(
        gradient_accumulation_steps
    )  # Batch of batches, helpful for gradient accumulation. Shape [grad_steps, batch_size, ...]
    # One iteration with 10 dupes, 8 nodes seems to be 60-70k steps.
    train_dataset = train_dataset.prefetch(buffer_size=8)

    # Validation should only be done on one node, since Horovod doesn't allow allreduce on a subset of ranks
    if hvd.rank() == 0:
        validation_dataset = get_mlm_dataset(
            filenames=validation_filenames,
            max_seq_length=max_seq_length,
            max_predictions_per_seq=max_predictions_per_seq,
            batch_size=batch_size,
        )
        # validation_dataset = validation_dataset.batch(1)
        validation_dataset = validation_dataset.prefetch(buffer_size=8)

        pbar = tqdm.tqdm(total_steps)
        summary_writer = None  # Only create a writer if we make it through a successful step
        logger.info(f"Starting training, job name {run_name}")

    i = 0
    for batch in train_dataset:
        learning_rate = schedule(step=tf.constant(i, dtype=tf.float32))
        loss_scale = opt.loss_scale()
        loss, mlm_loss, mlm_acc, sop_loss, sop_acc, grad_norm, weight_norm = train_step(
            model=model,
            opt=opt,
            gradient_accumulator=gradient_accumulator,
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            skip_sop=skip_sop,
            skip_mlm=skip_mlm,
        )

        # Don't want to wrap broadcast_variables() in a tf.function, can lead to asynchronous errors
        if i == 0:
            if hvd.rank() == 0 and loaded_opt_weights is not None:
                opt.set_weights(loaded_opt_weights)
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)
            i = opt.get_weights()[0] - 1

        is_final_step = i >= total_steps - 1
        do_squad = i in squad_steps or is_final_step
        # Squad requires all the ranks to train, but results are only returned on rank 0
        if do_squad:
            squad_results = get_squad_results_while_pretraining(
                model=model,
                model_size=model_size,
                fsx_prefix=fsx_prefix,
                step=i,
                fast=fast_squad,
                dummy_eval=dummy_eval,
            )
            if hvd.rank() == 0:
                squad_exact, squad_f1 = squad_results["exact"], squad_results["f1"]
                logger.info(f"SQuAD step {i} -- F1: {squad_f1:.3f}, Exact: {squad_exact:.3f}")
            # Re-wrap autograph so it doesn't get arg mismatches
            wrap_global_functions(do_gradient_accumulation)

        if hvd.rank() == 0:
            do_log = i % log_frequency == 0
            do_checkpoint = (i % checkpoint_frequency == 0) or is_final_step
            do_validation = (i % validate_frequency == 0) or is_final_step

            pbar.update(1)
            description = f"Loss: {loss:.3f}, MLM: {mlm_loss:.3f}, SOP: {sop_loss:.3f}, MLM_acc: {mlm_acc:.3f}, SOP_acc: {sop_acc:.3f}"
            pbar.set_description(description)
            if do_log:
                logger.info(f"Train step {i} -- {description}")

            if do_checkpoint:
                checkpoint_prefix = f"{fsx_prefix}/checkpoints/albert/{run_name}-step{i}"
                model_ckpt = f"{checkpoint_prefix}.ckpt"
                opt_ckpt = f"{checkpoint_prefix}-opt.npy"
                logger.info(f"Saving model at {model_ckpt}, optimizer at {opt_ckpt}")
                model.save_weights(model_ckpt)
                # model.load_weights(model_ckpt)

                opt_weights = opt.get_weights()
                np.save(opt_ckpt, opt_weights)
                # opt.set_weights(opt_weights)

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
                    f"{fsx_prefix}/logs/albert/{run_name}"
                )
            # Log to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar("weight_norm", weight_norm, step=i)
                tf.summary.scalar("loss_scale", loss_scale, step=i)
                tf.summary.scalar("learning_rate", learning_rate, step=i)
                tf.summary.scalar("train_loss", loss, step=i)
                tf.summary.scalar("train_mlm_loss", mlm_loss, step=i)
                tf.summary.scalar("train_mlm_acc", mlm_acc, step=i)
                tf.summary.scalar("train_sop_loss", sop_loss, step=i)
                tf.summary.scalar("train_sop_acc", sop_acc, step=i)
                tf.summary.scalar("grad_norm", grad_norm, step=i)
                if do_validation:
                    tf.summary.scalar("val_loss", val_loss, step=i)
                    tf.summary.scalar("val_mlm_loss", val_mlm_loss, step=i)
                    tf.summary.scalar("val_mlm_acc", val_mlm_acc, step=i)
                    tf.summary.scalar("val_sop_loss", val_sop_loss, step=i)
                    tf.summary.scalar("val_sop_acc", val_sop_acc, step=i)
                if do_squad:
                    tf.summary.scalar("squad_f1", squad_f1, step=i)
                    tf.summary.scalar("squad_exact", squad_exact, step=i)

        i += 1
        if is_final_step:
            break

    if hvd.rank() == 0:
        pbar.close()
        logger.info(f"Finished pretraining, job name {run_name}")


def get_squad_steps(extra_steps_str: str) -> List[int]:
    """ Parse a comma-separated string of integers, append it to list of default steps. """
    extra_squad_steps = [int(val) for val in extra_steps_str.split(",")] if extra_steps_str else []
    default_squad_steps = [
        k * 1000
        for k in [
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            120,
            140,
            160,
            180,
            200,
            220,
            240,
            260,
            280,
            300,
            320,
            340,
            360,
            380,
            400,
        ]
    ]
    return extra_squad_steps + default_squad_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Unused, but passed by SageMaker")
    parser.add_argument("--model_type", default="albert", choices=["albert", "bert"])
    parser.add_argument("--model_size", default="base", choices=["base", "large"])
    parser.add_argument("--batch_size", type=int, default=32, help="per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=512, choices=[128, 512])
    parser.add_argument("--warmup_steps", type=int, default=3125)
    parser.add_argument("--total_steps", type=int, default=125000)
    parser.add_argument("--learning_rate", type=float, default=0.00176)
    parser.add_argument("--end_learning_rate", type=float, default=3e-5)
    parser.add_argument("--learning_rate_decay_power", type=float, default=1.0)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", default="lamb", choices=["lamb", "adam"])
    parser.add_argument("--name", default="", help="Additional info to append to metadata")
    parser.add_argument(
        "--load_from", default="scratch", choices=["scratch", "checkpoint", "huggingface"],
    )
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument(
        "--fsx_prefix",
        default="/fsx",
        choices=["/fsx", "/opt/ml/input/data/training"],
        help="Change to /opt/ml/input/data/training on SageMaker",
    )
    # SageMaker does not work with 'store_const' args, since it parses into a dictionary
    # We will treat any value not equal to None as True, and use --skip_xla=true
    parser.add_argument(
        "--skip_xla",
        choices=["true"],
        help="For debugging. Faster startup time, slower runtime, more GPU vRAM.",
    )
    parser.add_argument(
        "--eager",
        choices=["true"],
        help="For debugging. Faster launch, slower runtime, more GPU vRAM.",
    )
    parser.add_argument(
        "--skip_sop", choices=["true"], help="Only use MLM loss, and exclude the SOP loss.",
    )
    parser.add_argument(
        "--skip_mlm", choices=["true"], help="Only use SOP loss, and exclude the MLM loss.",
    )
    parser.add_argument(
        "--pre_layer_norm",
        choices=["true"],
        help="Place layer normalization before the attention & FFN, rather than after adding the residual connection. https://openreview.net/pdf?id=B1x8anVFPr",
    )
    parser.add_argument("--extra_squad_steps", type=str)
    parser.add_argument("--fast_squad", choices=["true"])
    parser.add_argument("--dummy_eval", choices=["true"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tf.random.set_seed(args.seed)
    tf.autograph.set_verbosity(0)

    # Horovod init
    hvd.init()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    # XLA, AutoGraph
    parse_bool = lambda arg: arg == "true"
    tf.config.optimizer.set_jit(not parse_bool(args.skip_xla))
    tf.config.experimental_run_functions_eagerly(parse_bool(args.eager))

    main(
        fsx_prefix=args.fsx_prefix,
        load_from=args.load_from,
        checkpoint_path=args.checkpoint_path,
        model_type=args.model_type,
        model_size=args.model_size,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimizer=args.optimizer,
        name=args.name,
        learning_rate=args.learning_rate,
        end_learning_rate=args.end_learning_rate,
        learning_rate_decay_power=args.learning_rate_decay_power,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        skip_sop=parse_bool(args.skip_sop),
        skip_mlm=parse_bool(args.skip_mlm),
        pre_layer_norm=parse_bool(args.pre_layer_norm),
        fast_squad=parse_bool(args.fast_squad),
        dummy_eval=parse_bool(args.dummy_eval),
        squad_steps=get_squad_steps(args.extra_squad_steps),
        hidden_dropout_prob=args.hidden_dropout_prob,
        seed=args.seed,
    )
