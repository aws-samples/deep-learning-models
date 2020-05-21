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
import time
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

from common.arguments import populate_pretraining_parser
from common.datasets import get_mlm_dataset
from common.learning_rate_schedules import LinearWarmupPolyDecaySchedule
from common.utils import TqdmLoggingHandler, gather_indexes, rewrap_tf_function
from run_squad import get_squad_results_while_pretraining

# See https://github.com/huggingface/transformers/issues/3782; this import must come last
import horovod.tensorflow as hvd  # isort:skip

logger = logging.getLogger(__name__)


def get_squad_steps(extra_steps_str: str) -> List[int]:
    """ Parse a comma-separated string of integers, append it to list of default steps. """
    extra_squad_steps = [int(val) for val in extra_steps_str.split(",")] if extra_steps_str else []
    # fmt: off
    default_squad_steps = [
        k * 1000
        for k in [5, 10, 20,40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
    ]
    # fmt: on
    return extra_squad_steps + default_squad_steps


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
    (grads, grad_norm) = tf.clip_by_global_norm(grads, clip_norm=max_grad_norm)
    
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


def main():
    parser = argparse.ArgumentParser()
    populate_pretraining_parser(parser)
    args = parser.parse_args()
    tf.random.set_seed(args.seed)
    tf.autograph.set_verbosity(0)

    # Settings init
    parse_bool = lambda arg: arg == "true"
    checkpoint_frequency = 5000
    validate_frequency = 2000
    histogram_frequency = 100
    do_gradient_accumulation = args.gradient_accumulation_steps > 1
    do_xla = not parse_bool(args.skip_xla)
    do_eager = parse_bool(args.eager)
    skip_sop = parse_bool(args.skip_sop)
    skip_mlm = parse_bool(args.skip_mlm)
    pre_layer_norm = parse_bool(args.pre_layer_norm)
    fast_squad = parse_bool(args.fast_squad)
    dummy_eval = parse_bool(args.dummy_eval)
    squad_steps = get_squad_steps(args.extra_squad_steps)
    is_sagemaker = args.fsx_prefix.startswith("/opt/ml")
    disable_tqdm = is_sagemaker
    global max_grad_norm
    max_grad_norm = args.max_grad_norm

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

        metadata = (
            f"{args.model_type}"
            f"-{args.model_size}"
            f"-{args.load_from}"
            f"-{hvd.size()}gpus"
            f"-{args.batch_size}batch"
            f"-{args.gradient_accumulation_steps}accum"
            f"-{args.learning_rate}maxlr"
            f"-{args.end_learning_rate}endlr"
            f"-{args.learning_rate_decay_power}power"
            f"-{args.max_grad_norm}maxgrad"
            f"-{args.optimizer}opt"
            f"-{args.total_steps}steps"
            f"-{args.max_seq_length}seq"
            f"-{args.max_predictions_per_seq}preds"
            f"-{'preln' if pre_layer_norm else 'postln'}"
            f"{loss_str}"
            f"-{args.hidden_dropout_prob}dropout"
            f"-{args.seed}seed"
        )
        run_name = f"{current_time}-{platform}-{metadata}-{args.name if args.name else 'unnamed'}"

        # Logging should only happen on a single process
        # https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
        level = logging.INFO
        format = "%(asctime)-15s %(name)-12s: %(levelname)-8s %(message)s"
        handlers = [
            logging.FileHandler(f"{args.fsx_prefix}/logs/albert/{run_name}.log"),
            TqdmLoggingHandler(),
        ]
        logging.basicConfig(level=level, format=format, handlers=handlers)

        # Check that arguments passed in properly, only after registering the alert_func and logging
        assert not (skip_sop and skip_mlm), "Cannot use --skip_sop and --skip_mlm"

    wrap_global_functions(do_gradient_accumulation)

    if args.model_type == "albert":
        model_desc = f"albert-{args.model_size}-v2"
    elif args.model_type == "bert":
        model_desc = f"bert-{args.model_size}-uncased"

    config = AutoConfig.from_pretrained(model_desc)
    config.pre_layer_norm = pre_layer_norm
    config.hidden_dropout_prob = args.hidden_dropout_prob
    model = TFAutoModelForPreTraining.from_config(config)

    # Create optimizer and enable AMP loss scaling.
    schedule = LinearWarmupPolyDecaySchedule(
        max_learning_rate=args.learning_rate,
        end_learning_rate=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        power=args.learning_rate_decay_power,
    )
    if args.optimizer == "lamb":
        opt = LAMB(
            learning_rate=schedule,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )
    elif args.optimizer == "adam":
        opt = AdamW(weight_decay=0.0, learning_rate=schedule)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt, loss_scale="dynamic")
    gradient_accumulator = GradientAccumulator()

    loaded_opt_weights = None
    if args.load_from == "scratch":
        pass
    elif args.load_from.startswith("huggingface"):
        assert args.model_type == "albert", "Only loading pretrained albert models is supported"
        huggingface_name = f"albert-{args.model_size}-v2"
        if args.load_from == "huggingface":
            albert = TFAlbertModel.from_pretrained(huggingface_name, config=config)
            model.albert = albert
    else:
        model_ckpt, opt_ckpt = get_checkpoint_paths_from_prefix(args.checkpoint_path)

        model = TFAutoModelForPreTraining.from_config(config)
        if hvd.rank() == 0:
            model.load_weights(model_ckpt)
            loaded_opt_weights = np.load(opt_ckpt, allow_pickle=True)
            # We do not set the weights yet, we have to do a first step to initialize the optimizer.

    # Train filenames are [1, 2047], Val filenames are [0]. Note the different subdirectories
    # Move to same folder structure and remove if/else
    if args.model_type == "albert":
        train_glob = f"{args.fsx_prefix}/albert_pretraining/tfrecords/train/max_seq_len_{args.max_seq_length}_max_predictions_per_seq_{args.max_predictions_per_seq}_masked_lm_prob_15/albert_*.tfrecord"
        validation_glob = f"{args.fsx_prefix}/albert_pretraining/tfrecords/validation/max_seq_len_{args.max_seq_length}_max_predictions_per_seq_{args.max_predictions_per_seq}_masked_lm_prob_15/albert_*.tfrecord"
    if args.model_type == "bert":
        train_glob = f"{args.fsx_prefix}/bert_pretraining/max_seq_len_{args.max_seq_length}_max_predictions_per_seq_{args.max_predictions_per_seq}_masked_lm_prob_15/training/*.tfrecord"
        validation_glob = f"{args.fsx_prefix}/bert_pretraining/max_seq_len_{args.max_seq_length}_max_predictions_per_seq_{args.max_predictions_per_seq}_masked_lm_prob_15/validation/*.tfrecord"


    train_filenames = glob.glob(train_glob)
    validation_filenames = glob.glob(validation_glob)

    train_dataset = get_mlm_dataset(
        filenames=train_filenames,
        max_seq_length=args.max_seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
        batch_size=args.batch_size,
    )  # Of shape [batch_size, ...]
    # Batch of batches, helpful for gradient accumulation. Shape [grad_steps, batch_size, ...]
    train_dataset = train_dataset.batch(args.gradient_accumulation_steps)
    # One iteration with 10 dupes, 8 nodes seems to be 60-70k steps.
    train_dataset = train_dataset.prefetch(buffer_size=8)

    # Validation should only be done on one node, since Horovod doesn't allow allreduce on a subset of ranks
    if hvd.rank() == 0:
        validation_dataset = get_mlm_dataset(
            filenames=validation_filenames,
            max_seq_length=args.max_seq_length,
            max_predictions_per_seq=args.max_predictions_per_seq,
            batch_size=args.batch_size,
        )
        # validation_dataset = validation_dataset.batch(1)
        validation_dataset = validation_dataset.prefetch(buffer_size=8)

        pbar = tqdm.tqdm(args.total_steps, disable=disable_tqdm)
        summary_writer = None  # Only create a writer if we make it through a successful step
        logger.info(f"Starting training, job name {run_name}")

    i = 0
    start_time = time.perf_counter()
    for batch in train_dataset:
        learning_rate = schedule(step=tf.constant(i, dtype=tf.float32))
        loss_scale = opt.loss_scale()
        loss, mlm_loss, mlm_acc, sop_loss, sop_acc, grad_norm, weight_norm = train_step(
            model=model,
            opt=opt,
            gradient_accumulator=gradient_accumulator,
            batch=batch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
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

        is_final_step = i >= args.total_steps - 1
        do_squad = i in squad_steps or is_final_step
        # Squad requires all the ranks to train, but results are only returned on rank 0
        if do_squad:
            squad_results = get_squad_results_while_pretraining(
                model=model,
                model_size=args.model_size,
                fsx_prefix=args.fsx_prefix,
                step=i,
                fast=args.fast_squad,
                dummy_eval=args.dummy_eval,
            )
            if hvd.rank() == 0:
                squad_exact, squad_f1 = squad_results["exact"], squad_results["f1"]
                logger.info(f"SQuAD step {i} -- F1: {squad_f1:.3f}, Exact: {squad_exact:.3f}")
            # Re-wrap autograph so it doesn't get arg mismatches
            wrap_global_functions(do_gradient_accumulation)

        if hvd.rank() == 0:
            do_log = i % args.log_frequency == 0
            do_checkpoint = ((i > 0) and (i % checkpoint_frequency == 0)) or is_final_step
            do_validation = ((i > 0) and (i % validate_frequency == 0)) or is_final_step

            pbar.update(1)
            description = f"Loss: {loss:.3f}, MLM: {mlm_loss:.3f}, SOP: {sop_loss:.3f}, MLM_acc: {mlm_acc:.3f}, SOP_acc: {sop_acc:.3f}"
            pbar.set_description(description)
            if do_log:
                elapsed_time = time.perf_counter() - start_time
                if i == 0:
                    logger.info(f"First step: {elapsed_time:.3f} secs")
                else:
                    it_per_sec = args.log_frequency / elapsed_time
                    logger.info(f"Train step {i} -- {description} -- It/s: {it_per_sec:.2f}")
                    start_time = time.perf_counter()

            if do_checkpoint:
                checkpoint_prefix = f"{args.fsx_prefix}/checkpoints/albert/{run_name}-step{i}"
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
                    f"{args.fsx_prefix}/logs/albert/{run_name}"
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


if __name__ == "__main__":
    main()
