"""
# Tuple with keys (
# (input_ids, attention_mask, token_type_ids),
# (start_position, end_position, cls_index, p_mask, is_impossible)

g4dn can do evaluation w/ batch32 with 9GB memory. 1.08 it/s
1.61 it/s when performing inference on single-GPU.

Multi-gpu single-node throws an error, probably OOM.
"""


import argparse
import datetime
import logging
from typing import Dict, List, Optional, Union

import tensorflow as tf
import tensorflow_addons as tfa
import tqdm
from transformers import (
    AlbertTokenizer,
    AutoConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    TFAutoModelForQuestionAnswering,
    TFPreTrainedModel,
)
from transformers.data.processors.squad import (
    SquadExample,
    SquadFeatures,
    SquadProcessor,
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features,
)

from learning_rate_schedules import LinearWarmupLinearDecaySchedule
from models import load_qa_from_pretrained
from run_squad_evaluation import get_evaluation_metrics
from utils import f1_score, get_dataset, get_tokenizer

# See https://github.com/huggingface/transformers/issues/3782; this import must come last
import horovod.tensorflow as hvd  # isort:skip

logger = logging.getLogger(__name__)


def loss_fn(start_logits, end_logits, start_positions, end_positions, attention_mask):
    # Is this masking even necessary?
    # We need to mask so only the relevant logits contribute to crossentropy loss
    # Setting logits to -100 should make exp(-100) close to 0.
    mask = tf.math.equal(attention_mask, 1)
    start_logits_masked = tf.where(mask, start_logits, -100)
    end_logits_masked = tf.where(mask, end_logits, -100)

    start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=start_positions, logits=start_logits_masked
    )
    end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=end_positions, logits=end_logits_masked
    )
    total_loss = (start_loss + end_loss) / 2

    start_hat, end_hat = (
        tf.math.argmax(start_logits_masked, axis=-1),
        tf.math.argmax(end_logits_masked, axis=-1),
    )

    start_acc = tf.math.equal(start_hat, start_positions)  # tf.bool
    end_acc = tf.math.equal(end_hat, end_positions)
    total_acc = (tf.cast(start_acc, tf.float16) + tf.cast(end_acc, tf.float16)) / 2

    exact_match = tf.cast(tf.math.logical_and(start_acc, end_acc), tf.float16)

    # TF default to 0 being a tf.int32, which clashes with the int64 type of start_positions
    zero_int64 = tf.constant(0, dtype=tf.int64)
    zero_float64 = tf.constant(0, dtype=tf.float64)
    correct_tokens = tf.math.maximum(
        zero_int64,
        1 + tf.math.minimum(end_hat, end_positions) - tf.math.maximum(start_hat, start_positions),
    )
    total_tokens = 1 + end_positions - start_positions
    total_tokens_hat = tf.math.maximum(zero_int64, 1 + end_hat - start_hat)
    precision = tf.math.maximum(zero_float64, correct_tokens / total_tokens_hat)  # Convert NaN to 0
    recall = tf.math.maximum(zero_float64, correct_tokens / total_tokens)  # Convert NaN to 0

    # Reduce along batch dimension
    total_loss = tf.reduce_mean(total_loss)
    total_acc = tf.reduce_mean(total_acc)
    exact_match = tf.reduce_mean(exact_match)
    precision = tf.reduce_mean(precision)
    recall = tf.reduce_mean(recall)

    outputs = (total_loss, total_acc, exact_match, precision, recall)
    return outputs


def train_step(model, optimizer, batch) -> List[tf.Tensor]:
    with tf.GradientTape() as tape:
        input_dict = {
            "input_ids": batch[0]["input_ids"],
            "attention_mask": batch[0]["attention_mask"],
            "token_type_ids": batch[0]["token_type_ids"],
        }
        outputs = model(input_dict, training=True)
        start_logits, end_logits = outputs[0], outputs[1]
        loss, acc, exact_match, precision, recall = loss_fn(
            start_logits=start_logits,
            end_logits=end_logits,
            start_positions=batch[1]["start_position"],
            end_positions=batch[1]["end_position"],
            attention_mask=batch[0]["attention_mask"],
        )
        scaled_loss = optimizer.get_scaled_loss(loss)

    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)

    allreduce_grads = [hvd.allreduce(grad) if grad is not None else None for grad in grads]

    (clipped_grads, grad_norm) = tf.clip_by_global_norm(allreduce_grads, clip_norm=1.0)
    non_none_grads = [
        (grad, var)
        for (grad, var) in zip(clipped_grads, model.trainable_variables)
        if grad is not None
    ]
    optimizer.apply_gradients(non_none_grads)

    loss = hvd.allreduce(loss)
    acc = hvd.allreduce(acc)
    exact_match = hvd.allreduce(exact_match)
    precision = hvd.allreduce(precision)
    recall = hvd.allreduce(recall)
    f1 = f1_score(precision=precision, recall=recall)

    return loss, acc, exact_match, f1, precision, recall


def validation_step(model, batch) -> List[tf.Tensor]:
    input_dict = {
        "input_ids": batch[0]["input_ids"],
        "attention_mask": batch[0]["attention_mask"],
        "token_type_ids": batch[0]["token_type_ids"],
    }
    outputs = model(input_dict, training=False)
    start_logits, end_logits = outputs[0], outputs[1]
    loss, acc, exact_match, precision, recall = loss_fn(
        start_logits=start_logits,
        end_logits=end_logits,
        start_positions=batch[1]["start_position"],
        end_positions=batch[1]["end_position"],
        attention_mask=batch[0]["attention_mask"],
    )
    return loss, acc, exact_match, precision, recall


def run_validation(model, val_dataset, num_batches: int = 100) -> List[tf.Tensor]:
    wrapped_validation_step = tf.function(validation_step)
    val_loss, val_acc, val_exact_match, val_precision, val_recall = (0, 0, 0, 0, 0)
    for batch in val_dataset.take(num_batches):
        loss, acc, exact_match, precision, recall = wrapped_validation_step(model, batch)
        val_loss += loss
        val_acc += acc
        val_exact_match += exact_match
        val_precision += precision
        val_recall += recall

    val_loss /= num_batches
    val_acc /= num_batches
    val_exact_match /= num_batches
    val_precision /= num_batches
    val_recall /= num_batches
    val_f1 = f1_score(val_precision, val_recall)

    return (val_loss, val_acc, val_exact_match, val_f1, val_precision, val_recall)


def print_eval_metrics(results, step) -> None:
    """ Print evaluation metrics to console. """
    description = (
        f"Step {step} evaluation - EM: {results['exact']:.3f}, F1: {results['f1']:.3f}, "
        f"HasAnsEM: {results['HasAns_exact']:.3f}, HasAnsF1: {results['HasAns_f1']:.3f}, "
        f"NoAnsEM: {results['NoAns_exact']:.3f}, NoAnsF1: {results['NoAns_f1']:.3f}\n"
    )
    print(description)


def tensorboard_eval_metrics(summary_writer, results: Dict, step: int) -> None:
    """ Log evaluation metrics to TensorBoard. """
    tf.summary.scalar("eval_exact", results["exact"], step=step)
    tf.summary.scalar("eval_f1", results["f1"], step=step)
    tf.summary.scalar("eval_hasans_exact", results["HasAns_exact"], step=step)
    tf.summary.scalar("eval_hasans_f1", results["HasAns_f1"], step=step)
    tf.summary.scalar("eval_noans_exact", results["NoAns_exact"], step=step)
    tf.summary.scalar("eval_noans_f1", results["NoAns_f1"], step=step)


def wrap_tf_function_idempotent(func):
    if hasattr(func, "python_function"):
        return func
    else:
        return tf.function(func)


def run_squad_and_get_results(
    run_name: str,
    fsx_prefix: str,
    pre_layer_norm: bool,
    model_size: str,
    load_from: Union[str, tf.keras.Model],
    load_step: int,
    batch_size: int,
    checkpoint_frequency: Optional[int],
    validate_frequency: Optional[int],
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    dataset: str,
    dummy_eval: bool = False,
    config: Optional[PretrainedConfig] = None,
) -> Dict:
    checkpoint_frequency = checkpoint_frequency or 1000000
    validate_frequency = validate_frequency or 1000000

    if isinstance(load_from, tf.keras.Model):
        config = load_from.config
    assert config is not None, "config may not be None"

    # Instantiate QuestionAnswering model
    if isinstance(load_from, TFPreTrainedModel):
        model = load_qa_from_pretrained(model=load_from)
    elif load_from == "scratch":
        model = TFAutoModelForQuestionAnswering.from_config(config)
    elif load_from == "huggingface":
        model = load_qa_from_pretrained(name=f"albert-{model_size}-v2")
    else:
        raise ValueError(
            f"'load_from' is '{load_from}'; must be in ['scratch', 'huggingface', 'amazon']"
        )

    tokenizer = get_tokenizer()

    schedule = LinearWarmupLinearDecaySchedule(
        max_learning_rate=learning_rate,
        end_learning_rate=0,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    optimizer = tfa.optimizers.AdamW(weight_decay=0.0, learning_rate=schedule)
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
        optimizer, loss_scale="dynamic"
    )

    model.call = wrap_tf_function_idempotent(model.call)

    if dataset == "squadv1":
        train_filename = "train-v1.1.json"
        val_filename = "dev-v1.1.json"
        processor = SquadV1Processor()
    elif dataset == "squadv2":
        train_filename = "train-v2.0.json"
        val_filename = "dev-v2.0.json"
        processor = SquadV2Processor()
    elif dataset == "debug":
        train_filename = "dev-v2.0.json"
        val_filename = "dev-v2.0.json"
        processor = SquadV2Processor()
    else:
        assert False, "--dataset must be one of ['squadv1', 'squadv2', 'debug']"

    data_dir = f"{fsx_prefix}/squad_data"

    train_dataset = get_dataset(
        tokenizer=tokenizer,
        processor=processor,
        data_dir=data_dir,
        filename=train_filename,
        batch_size=batch_size,
        shard=True,
        shuffle=True,
        repeat=True,
        drop_remainder=True,
    )

    if hvd.rank() == 0:
        print("Starting finetuning")
        pbar = tqdm.tqdm(total_steps)
        summary_writer = None  # Only create a writer if we make it through a successful step
        val_dataset = get_dataset(
            tokenizer=tokenizer,
            processor=processor,
            data_dir=data_dir,
            filename=val_filename,
            batch_size=batch_size,
            shard=False,
            shuffle=True,
            drop_remainder=False,
        )

    # Need to re-wrap every time this function is called
    # Wrapping train_step gives an error with optimizer initialization on the second pass
    # of run_squad_and_get_results(). Bug report at https://github.com/tensorflow/tensorflow/issues/38875
    # Discussion at https://github.com/tensorflow/tensorflow/issues/27120
    wrapped_train_step = tf.function(train_step)
    for step, batch in enumerate(train_dataset):
        learning_rate = schedule(step=tf.constant(step, dtype=tf.float32))
        loss, acc, exact_match, f1, precision, recall = wrapped_train_step(
            model=model, optimizer=optimizer, batch=batch
        )

        # Broadcast model after the first step so parameters and optimizer are initialized
        if step == 0:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        is_final_step = step >= total_steps - 1
        if hvd.rank() == 0:
            do_checkpoint = (step % checkpoint_frequency == 0) or is_final_step
            do_validate = (step % validate_frequency == 0) or is_final_step

            pbar.update(1)
            description = f"Loss: {loss:.3f}, Acc: {acc:.3f}, EM: {exact_match:.3f}, F1: {f1:.3f}"
            pbar.set_description(description)

            if do_validate:
                print("Running validation")
                (
                    val_loss,
                    val_acc,
                    val_exact_match,
                    val_f1,
                    val_precision,
                    val_recall,
                ) = run_validation(model=model, val_dataset=val_dataset)
                description = (
                    f"Step {step} validation - Loss: {val_loss:.3f}, Acc: {val_acc:.3f}, "
                    f"EM: {val_exact_match:.3f}, F1: {val_f1:.3f}"
                )
                print(description)
                print("Running evaluation")
                if dummy_eval:
                    results = {
                        "exact": 0.8169797018445212,
                        "f1": 4.4469722448269335,
                        "total": 11873,
                        "HasAns_exact": 0.15182186234817813,
                        "HasAns_f1": 7.422216845956518,
                        "HasAns_total": 5928,
                        "NoAns_exact": 1.4802354920100924,
                        "NoAns_f1": 1.4802354920100924,
                        "NoAns_total": 5945,
                        "best_exact": 50.07159100480081,
                        "best_exact_thresh": 0.0,
                        "best_f1": 50.0772059855695,
                        "best_f1_thresh": 0.0,
                    }
                else:
                    results: Dict = get_evaluation_metrics(
                        model=model, data_dir=data_dir, filename=val_filename, batch_size=32,
                    )
                print_eval_metrics(results=results, step=step)

            if do_checkpoint:
                checkpoint_path = (
                    f"{fsx_prefix}/checkpoints/albert-squad/{run_name}-step{step}.ckpt"
                )
                print(f"Saving checkpoint at {checkpoint_path}")
                model.save_weights(checkpoint_path)

            if summary_writer is None:
                summary_writer = tf.summary.create_file_writer(
                    f"{fsx_prefix}/logs/albert-squad/{run_name}"
                )
            with summary_writer.as_default():
                tf.summary.scalar("learning_rate", learning_rate, step=step)
                tf.summary.scalar("train_loss", loss, step=step)
                tf.summary.scalar("train_acc", acc, step=step)
                tf.summary.scalar("train_exact", exact_match, step=step)
                tf.summary.scalar("train_f1", f1, step=step)
                tf.summary.scalar("train_precision", precision, step=step)
                tf.summary.scalar("train_recall", recall, step=step)
                if do_validate:
                    tf.summary.scalar("val_loss", val_loss, step=step)
                    tf.summary.scalar("val_acc", val_acc, step=step)
                    tf.summary.scalar("val_exact", val_exact_match, step=step)
                    tf.summary.scalar("val_f1", val_f1, step=step)
                    tf.summary.scalar("val_precision", val_precision, step=step)
                    tf.summary.scalar("val_recall", val_recall, step=step)
                    # And the eval metrics
                    tensorboard_eval_metrics(
                        summary_writer=summary_writer, results=results, step=step
                    )

        if is_final_step:
            break

    # Can we return a value only on a single rank?
    if hvd.rank() == 0:
        pbar.close()
        print(f"Finished finetuning, job name {run_name}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model loading
    parser.add_argument("--model_type", default="albert", choices=["albert", "bert"])
    parser.add_argument("--model_size", default="base", choices=["base", "large"])
    parser.add_argument("--load_from", required=True)
    parser.add_argument("--load_step", type=int)
    parser.add_argument("--skip_amp", choices=["true"])
    parser.add_argument("--skip_xla", choices=["true"])
    parser.add_argument("--eager", choices=["true"])
    parser.add_argument(
        "--pre_layer_norm",
        choices=["true"],
        help="See https://github.com/huggingface/transformers/pull/3929",
    )
    parser.add_argument(
        "--fsx_prefix",
        default="/fsx",
        choices=["/fsx", "/opt/ml/input/data/training"],
        help="Change to /opt/ml/input/data/training on SageMaker",
    )
    # Hyperparameters from https://arxiv.org/pdf/1909.11942.pdf#page=17
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--total_steps", default=8144, type=int)
    parser.add_argument("--warmup_steps", default=814, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--dataset", default="squadv2")
    # Logging information
    parser.add_argument("--name", default="default")
    parser.add_argument("--validate_frequency", default=1000, type=int)
    parser.add_argument("--checkpoint_frequency", default=500, type=int)
    parser.add_argument("--model_dir", help="Unused, but passed by SageMaker")
    args = parser.parse_args()
    tf.random.set_seed(42)

    # Horovod init
    hvd.init()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    # XLA, AMP, AutoGraph
    parse_bool = lambda arg: arg == "true"
    tf.config.optimizer.set_jit(not parse_bool(args.skip_xla))
    tf.config.optimizer.set_experimental_options(
        {"auto_mixed_precision": not parse_bool(args.skip_amp)}
    )
    tf.config.experimental_run_functions_eagerly(parse_bool(args.eager))

    if hvd.rank() == 0:
        # Run name should only be used on one process to avoid race conditions
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        platform = "eks" if args.fsx_prefix == "/fsx" else "sm"
        if args.load_from.startswith("amazon"):
            load_name = f"{args.load_from}{args.load_step}"
        else:
            load_name = args.load_from
        run_name = f"{current_time}-{platform}-{args.model_size}-{args.dataset}-{load_name}-{hvd.size()}gpus-{args.batch_size}batch-{args.learning_rate}lr-{args.name}"
    else:
        # We only use run_name on rank 0, but need all ranks to pass a value in function args
        run_name = None

    if args.model_type == "albert":
        model_desc = f"albert-{args.model_size}-v2"
    else:
        model_desc = f"bert-{args.model_size}-uncased"

    results = run_squad_and_get_results(
        run_name=run_name,
        fsx_prefix=args.fsx_prefix,
        pre_layer_norm=parse_bool(args.pre_layer_norm),
        model_size=args.model_size,
        load_from=args.load_from,
        load_step=args.load_step,
        batch_size=args.batch_size,
        checkpoint_frequency=args.checkpoint_frequency,
        validate_frequency=args.validate_frequency,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        dataset=args.dataset,
        config=AutoConfig.from_pretrained(model_desc),
    )
    if hvd.rank() == 0:
        print(results)
