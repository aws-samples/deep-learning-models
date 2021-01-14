"""
# Tuple with keys (
# (input_ids, attention_mask, token_type_ids),
# (start_position, end_position, cls_index, p_mask, is_impossible)

g4dn can do evaluation w/ batch32 with 9GB memory. 1.08 it/s
1.61 it/s when performing inference on single-GPU.

Multi-gpu single-node throws an error, probably OOM.
"""


import datetime
import logging
import math
import os
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import tensorflow as tf
import tensorflow_addons as tfa
import tqdm
from transformers import (
    AutoConfig,
    HfArgumentParser,
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
)

from albert.run_squad_evaluation import get_evaluation_metrics
from common.arguments import (
    DataTrainingArguments,
    LoggingArguments,
    ModelArguments,
    PathArguments,
    TrainingArguments,
)
from common.learning_rate_schedules import LinearWarmupPolyDecaySchedule
from common.models import create_model, load_qa_from_pretrained
from common.utils import (
    TqdmLoggingHandler,
    create_tokenizer,
    f1_score,
    get_dataset,
    rewrap_tf_function,
)

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
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        loss, acc, exact_match, precision, recall = loss_fn(
            start_logits=start_logits,
            end_logits=end_logits,
            start_positions=batch[1]["start_positions"],
            end_positions=batch[1]["end_positions"],
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
    start_logits, end_logits = outputs.start_logits, outputs.end_logits
    loss, acc, exact_match, precision, recall = loss_fn(
        start_logits=start_logits,
        end_logits=end_logits,
        start_positions=batch[1]["start_positions"],
        end_positions=batch[1]["end_positions"],
        attention_mask=batch[0]["attention_mask"],
    )
    return loss, acc, exact_match, precision, recall


def run_validation(model, val_dataset, num_batches: int = 100) -> List[tf.Tensor]:
    global validation_step
    validation_step = rewrap_tf_function(validation_step)
    val_loss, val_acc, val_exact_match, val_precision, val_recall = (0, 0, 0, 0, 0)
    for batch in val_dataset.take(num_batches):
        loss, acc, exact_match, precision, recall = validation_step(model, batch)
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
    val_f1 = f1_score(precision=val_precision, recall=val_recall)

    return (val_loss, val_acc, val_exact_match, val_f1, val_precision, val_recall)


def print_eval_metrics(results, step, dataset) -> None:
    """ Print evaluation metrics to console. """
    if dataset == "squadv2":
        description = (
            f"Step {step} evaluation - EM: {results['exact']:.3f}, F1: {results['f1']:.3f}, "
            f"HasAnsEM: {results['HasAns_exact']:.3f}, HasAnsF1: {results['HasAns_f1']:.3f}, "
            f"NoAnsEM: {results['NoAns_exact']:.3f}, NoAnsF1: {results['NoAns_f1']:.3f}\n"
        )
    else:
        description = (
            f"Step {step} evaluation - EM: {results['exact']:.3f}, F1: {results['f1']:.3f}, "
            f"HasAnsEM: {results['HasAns_exact']:.3f}, HasAnsF1: {results['HasAns_f1']:.3f}\n "
        )
    logger.info(description)


def tensorboard_eval_metrics(summary_writer, results: Dict, step: int, dataset) -> None:
    """ Log evaluation metrics to TensorBoard. """
    tf.summary.scalar("eval_exact", results["exact"], step=step)
    tf.summary.scalar("eval_f1", results["f1"], step=step)
    tf.summary.scalar("eval_hasans_exact", results["HasAns_exact"], step=step)
    tf.summary.scalar("eval_hasans_f1", results["HasAns_f1"], step=step)
    if dataset == "squadv2":
        tf.summary.scalar("eval_noans_exact", results["NoAns_exact"], step=step)
        tf.summary.scalar("eval_noans_f1", results["NoAns_f1"], step=step)


def hvd_barrier():
    hvd.allreduce(tf.random.normal([1]))


def get_squad_results_while_pretraining(
    model: tf.keras.Model,
    tokenizer: PreTrainedTokenizer,
    model_size: str,
    filesystem_prefix: str,
    step: int,
    dataset: str,
    fast: bool = False,
    dummy_eval: bool = False,
):
    # This is inefficient, since each rank will save and serialize the model separately.
    # It would be better to have rank 0 save the model and all the ranks read it, but
    # `run_name` isn't deterministic due to timestamps, so only rank 0 has the run_name.
    # TODO: Improve. If only tf.keras.clone_model(model) worked.
    with TemporaryDirectory() as dirname:
        path = os.path.join(dirname, "model")
        model.save_weights(path)
        hvd_barrier()
        # Convert model into a clone
        cloned_model = type(model)(config=model.config)
        cloned_model.load_weights(path).expect_partial()
        qa_model = load_qa_from_pretrained(model=cloned_model)
        qa_model.call = rewrap_tf_function(qa_model.call)
        #
        hvd_barrier()
        per_gpu_batch_size = min(3, int(math.ceil(48 / hvd.size())))
        if fast:
            warmup_steps = 5
            total_steps = 10
            dataset = "debug"
        if dataset == "squadv2":
            warmup_steps = 814
            total_steps = 8144
            learning_rate = 3e-5
        elif dataset == "squadv1":
            warmup_steps = 365
            total_steps = 3649
            learning_rate = 5e-5
        else:
            warmup_steps = 5
            total_steps = 10

        squad_run_name = f"pretrain{step}-"
        squad_results = run_squad_and_get_results(
            model=qa_model,
            tokenizer=tokenizer,
            run_name=squad_run_name,
            filesystem_prefix=filesystem_prefix,
            per_gpu_batch_size=per_gpu_batch_size,  # This will be less than 3, so no OOM errors
            checkpoint_frequency=None,
            validate_frequency=None,
            evaluate_frequency=None,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            dataset=dataset,
            dummy_eval=dummy_eval,
        )
        del cloned_model
        del qa_model
        hvd_barrier()

    if hvd.rank() == 0:
        return squad_results


def run_squad_and_get_results(
    model: tf.keras.Model,  # Must be QuestionAnswering model, not PreTraining
    tokenizer: PreTrainedTokenizer,
    run_name: str,
    filesystem_prefix: str,
    per_gpu_batch_size: int,
    checkpoint_frequency: Optional[int],
    validate_frequency: Optional[int],
    evaluate_frequency: Optional[int],
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    dataset: str,
    dummy_eval: bool = False,
) -> Dict:
    checkpoint_frequency = checkpoint_frequency or 1000000
    validate_frequency = validate_frequency or 1000000
    evaluate_frequency = evaluate_frequency or 1000000
    is_sagemaker = filesystem_prefix.startswith("/opt/ml")
    disable_tqdm = is_sagemaker

    schedule = LinearWarmupPolyDecaySchedule(
        max_learning_rate=learning_rate,
        end_learning_rate=0,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    optimizer = tfa.optimizers.AdamW(weight_decay=0.0, learning_rate=schedule)
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
        optimizer, loss_scale="dynamic"
    )  # AMP

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

    data_dir = os.path.join(filesystem_prefix, "squad_data")

    train_dataset = get_dataset(
        tokenizer=tokenizer,
        processor=processor,
        data_dir=data_dir,
        filename=train_filename,
        per_gpu_batch_size=per_gpu_batch_size,
        shard=True,
        shuffle=True,
        repeat=True,
        drop_remainder=True,
    )

    if hvd.rank() == 0:
        logger.info(f"Starting finetuning on {dataset}")
        pbar = tqdm.tqdm(total_steps, disable=disable_tqdm)
        summary_writer = None  # Only create a writer if we make it through a successful step
        val_dataset = get_dataset(
            tokenizer=tokenizer,
            processor=processor,
            data_dir=data_dir,
            filename=val_filename,
            per_gpu_batch_size=per_gpu_batch_size,
            shard=False,
            shuffle=True,
            drop_remainder=False,
        )

    # Need to re-wrap every time this function is called
    # Wrapping train_step gives an error with optimizer initialization on the second pass
    # of run_squad_and_get_results(). Bug report at https://github.com/tensorflow/tensorflow/issues/38875
    # Discussion at https://github.com/tensorflow/tensorflow/issues/27120
    global train_step
    train_step = rewrap_tf_function(train_step)

    for step, batch in enumerate(train_dataset):
        learning_rate = schedule(step=tf.constant(step, dtype=tf.float32))
        loss, acc, exact_match, f1, precision, recall = train_step(
            model=model, optimizer=optimizer, batch=batch
        )

        # Broadcast model after the first step so parameters and optimizer are initialized
        if step == 0:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        is_final_step = step >= total_steps - 1
        if hvd.rank() == 0:
            do_checkpoint = ((step > 0) and step % checkpoint_frequency == 0) or is_final_step
            do_validate = ((step > 0) and step % validate_frequency == 0) or is_final_step
            do_evaluate = ((step > 0) and step % evaluate_frequency == 0) or is_final_step

            pbar.update(1)
            description = f"Loss: {loss:.3f}, Acc: {acc:.3f}, EM: {exact_match:.3f}, F1: {f1:.3f}"
            pbar.set_description(description)

            if do_validate:
                logger.info("Running validation")
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
                logger.info(description)

            if do_evaluate:
                logger.info("Running evaluation")
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
                        model=model,
                        tokenizer=tokenizer,
                        data_dir=data_dir,
                        filename=val_filename,
                        per_gpu_batch_size=32,
                    )
                print_eval_metrics(results=results, step=step, dataset=dataset)

            if do_checkpoint:
                # TODO: Abstract out to specify any checkpoint path
                checkpoint_path = os.path.join(
                    filesystem_prefix, f"checkpoints/squad/{run_name}-step{step}.ckpt"
                )
                logger.info(f"Saving checkpoint at {checkpoint_path}")
                model.save_weights(checkpoint_path)

            if summary_writer is None:
                # TODO: Abstract out to specify any logs path
                summary_writer = tf.summary.create_file_writer(
                    os.path.join(filesystem_prefix, f"logs/squad/{run_name}")
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
                        summary_writer=summary_writer, results=results, step=step, dataset=dataset
                    )

        if is_final_step:
            break
    del train_dataset

    # Can we return a value only on a single rank?
    if hvd.rank() == 0:
        pbar.close()
        logger.info(f"Finished finetuning, job name {run_name}")
        return results


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments, PathArguments)
    )
    model_args, data_args, train_args, log_args, path_args = parser.parse_args_into_dataclasses()

    tf.random.set_seed(train_args.seed)
    tf.autograph.set_verbosity(0)

    level = logging.INFO
    format = "%(asctime)-15s %(name)-12s: %(levelname)-8s %(message)s"
    handlers = [
        TqdmLoggingHandler(),
    ]
    logging.basicConfig(level=level, format=format, handlers=handlers)

    # Horovod init
    hvd.init()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    # XLA, AMP, AutoGraph
    parse_bool = lambda arg: arg == "true"
    tf.config.optimizer.set_jit(not parse_bool(train_args.skip_xla))
    tf.config.experimental_run_functions_eagerly(parse_bool(train_args.eager))

    if hvd.rank() == 0:
        # Run name should only be used on one process to avoid race conditions
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        platform = "eks" if path_args.filesystem_prefix == "/fsx" else "sm"
        if log_args.run_name is None:
            run_name = f"{current_time}-{platform}-{model_args.model_type}-{model_args.model_size}-{data_args.squad_version}-{model_args.load_from}-{hvd.size()}gpus-{train_args.name}"
        else:
            run_name = log_args.run_name
    else:
        # We only use run_name on rank 0, but need all ranks to pass a value in function args
        run_name = None

    if model_args.load_from == "huggingface":
        logger.info(f"Loading weights from Huggingface {model_args.model_desc}")
        model = TFAutoModelForQuestionAnswering.from_pretrained(model_args.model_desc)
    else:
        model = create_model(model_class=TFAutoModelForQuestionAnswering, model_args=model_args)

    model.call = rewrap_tf_function(model.call)
    tokenizer = create_tokenizer(model_args.model_type)

    loaded_optimizer_weights = None
    if model_args.load_from == "checkpoint":
        if hvd.rank() == 0:
            checkpoint_path = os.path.join(path_args.filesystem_prefix, model_args.checkpoint_path)
            logger.info(f"Loading weights from {checkpoint_path}.ckpt")
            model.load_weights(f"{checkpoint_path}.ckpt").expect_partial()

    results = run_squad_and_get_results(
        model=model,
        tokenizer=tokenizer,
        run_name=run_name,
        filesystem_prefix=path_args.filesystem_prefix,
        per_gpu_batch_size=train_args.per_gpu_batch_size,
        checkpoint_frequency=log_args.checkpoint_frequency,
        validate_frequency=log_args.validation_frequency,
        evaluate_frequency=log_args.evaluate_frequency,
        learning_rate=train_args.learning_rate,
        warmup_steps=train_args.warmup_steps,
        total_steps=train_args.total_steps,
        dataset=data_args.squad_version,
    )
    if hvd.rank() == 0:
        logger.info(results)


if __name__ == "__main__":
    main()
