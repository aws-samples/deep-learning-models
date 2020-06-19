"""
Batch sizes: 32 = 5GB memory, 128 = 17GB

What's the best way to develop a new pretraining script?

Dynamic masking straight from text.
Abtract out the gradient accumulation functionality. Tracking loss, acc variables within the accumulator rather than outside.
Incorporate the new transformers version. Be willing to lose my current work.

# TODO: Should we include special tokens? <BOS>, <EOS>.
# TODO: Weight sharing between generator and discriminator, only token embeddings.
"""

import logging

import numpy as np
import tensorflow as tf
from transformers import (
    ElectraConfig,
    ElectraTokenizer,
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


@tf.function
def train_step(optimizer, gen, dis, ids, masked_ids, mask):
    with tf.GradientTape() as tape:

        (adv_logits,) = gen(masked_ids)  # [1, 6, 30522]
        truth = tf.boolean_mask(ids, mask)  # [4]
        preds = tf.boolean_mask(adv_logits, mask)  # [4, 30522] -> flattens the batch dimension
        gen_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=truth, y_pred=preds, from_logits=True
        )  # [4]
        gen_loss = tf.reduce_mean(gen_loss)

        adv_ids = tf.argmax(adv_logits, axis=-1)  # [1, 6]

        ids_equal = tf.cast(adv_ids == ids, dtype=tf.int64)
        gen_correct = tf.boolean_mask(ids_equal, mask)
        gen_acc = tf.reduce_mean(tf.cast(gen_correct, dtype=tf.float32))

        gen_ids = mask * adv_ids + (1 - mask) * ids

        (dis_logits,) = dis(gen_ids)  # [6], logits that
        # Linear layer is already in TFElectraDiscriminatorPredictions.
        dis_probs = tf.math.sigmoid(dis_logits)
        dis_preds = tf.cast(dis_probs > 0.5, dtype=mask.dtype)

        # If generator generates correct token, invert the loss
        dis_loss = tf.keras.losses.binary_crossentropy(
            y_true=tf.cast(gen_ids != ids, tf.int64), y_pred=dis_logits, from_logits=True
        )
        dis_loss = tf.reduce_mean(dis_loss)
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

    # TODO: Should I use bert-base-uncased?
    tokenizer = ElectraTokenizer.from_pretrained("bert-base-uncased")

    gen_config = ElectraConfig.from_pretrained("google/electra-small-generator")
    dis_config = ElectraConfig.from_pretrained("google/electra-small-discriminator")

    # gen = TFElectraForMaskedLM.from_pretrained("google/electra-small-generator")
    # dis = TFElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
    gen = TFElectraForMaskedLM(config=gen_config)
    dis = TFElectraForPreTraining(config=dis_config)
    optimizer = get_adamw_optimizer(train_args)
    # optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    # Load in WikiText-2.
    filename = "/fsx/wikitext/wikitext-2-raw/wiki.test.raw"
    with open(filename) as infile:
        wiki_text: str = infile.read()  # length 1,288,556

    # Load in text strings.
    text = "The chef cooked the meal. It was delicious and appetizing, yet I couldn't shake the feeling that Michael Jordan would have the flu game."
    # Convert to text tokens
    tokens = tokenizer.tokenize(text)  # ['the', 'chef', 'cooked', 'the', 'meal', '.']

    # Convert to token ids.
    ids = tokenizer.convert_tokens_to_ids(tokens)

    wiki_tokens = tokenizer.tokenize(wiki_text)  # length 273,178, tokenized instantaneously
    wiki_ids = tokenizer.convert_tokens_to_ids(wiki_tokens)

    wandb_run_name = None
    if hvd.rank() == 0:
        # disable_tqdm = False
        # pbar = tqdm.tqdm(train_args.total_steps, disable=disable_tqdm)
        # Logging should only happen on a single process
        # https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
        level = logging.INFO
        format = "%(asctime)-15s %(name)-12s: %(levelname)-8s %(message)s"
        handlers = [
            # logging.FileHandler(f"{data_args.fsx_prefix}/logs/electra/{run_name}.log"),
            TqdmLoggingHandler(),
        ]
        logging.basicConfig(level=level, format=format, handlers=handlers)

    for step in range(train_args.total_steps):
        bsz = train_args.per_gpu_batch_size
        seq_len = data_args.max_seq_length
        ids = tf.constant(
            select_batch_ids(arr=wiki_ids, bsz=bsz, seq_len=seq_len)
        )  # [bsz, seq_len]
        # Generate a mask.
        # Mask should be a boolean array where 1 represents masked token.
        mask_prob = 0.15
        mask = np.array(np.random.rand(*ids.shape) > 1 - mask_prob, dtype=int)
        tf_mask = tf.constant(mask)
        # Mask the token ids.
        masked_ids = np.where(mask, tokenizer.mask_token_id, ids)
        masked_ids = tf.constant(masked_ids)

        loss, gen_loss, dis_loss, gen_acc, dis_acc, gen_ids, dis_preds = train_step(
            optimizer=optimizer, gen=gen, dis=dis, ids=ids, masked_ids=masked_ids, mask=tf_mask
        )

        if step == 0:
            # Horovod broadcast
            hvd.broadcast_variables(dis.variables, root_rank=0)
            hvd.broadcast_variables(gen.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)
            # WandB init
            if is_wandb_available():
                config = {
                    "global_batch_size": hvd.size() * train_args.per_gpu_batch_size,
                    "per_gpu_batch_size": train_args.per_gpu_batch_size,
                    "max_seq_length": data_args.max_seq_length,
                }
                wandb.init(config=config, project="electra")
                wandb.run.save()
                wandb_run_name = wandb.run.name

        description = f"Step {step} -- gen_loss: {gen_loss:.3f}, dis_loss: {dis_loss:.3f}, gen_acc: {gen_acc:.3f}, dis_acc: {dis_acc:.3f}\n"
        if step % log_args.log_frequency == 0:
            logger.info(f"Original:            '{tokenizer.decode(ids[0].numpy())}'")
            logger.info(f"Masked:              '{tokenizer.decode(masked_ids[0].numpy())}'")
            logger.info(
                f"Generator output:    '{colorize_gen(tokenizer, ids[0], gen_ids[0], tf_mask[0])}'"
            )
            logger.info(
                f"Discriminator preds: '{colorize_dis(tokenizer, gen_ids[0], dis_preds[0])}'"
            )
            logger.info(description)

        # pbar.update(1)
        # pbar.set_description("hello")
        # pbar.set_description(description)

        train_metrics = {
            "train/loss": loss,
            "train/gen_loss": gen_loss,
            "train/dis_loss": dis_loss,
            "train/gen_acc": gen_acc,
            "train/dis_acc": dis_acc,
        }

        if is_wandb_available():
            wandb.log({"step": step, **train_metrics})


if __name__ == "__main__":
    main()
