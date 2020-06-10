"""
What's the best way to develop a new pretraining script?

Dynamic masking straight from text.
Abtract out the gradient accumulation functionality. Tracking loss, acc variables within the accumulator rather than outside.
Incorporate the new transformers version. Be willing to lose my current work.

# TODO: Should we include special tokens? <BOS>, <EOS>.
# TODO: Weight sharing between generator and discriminator, only token embeddings.

"""

import os

import numpy as np
import tensorflow as tf
from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    TFElectraForMaskedLM,
    TFElectraForPreTraining,
)

from electra.utils import colorize_dis, colorize_gen

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# TODO: Should I use bert-base-uncased?
tokenizer = ElectraTokenizer.from_pretrained("bert-base-uncased")

gen_config = ElectraConfig.from_pretrained("google/electra-small-generator")
dis_config = ElectraConfig.from_pretrained("google/electra-small-discriminator")

gen = TFElectraForMaskedLM.from_pretrained("google/electra-small-generator")
dis = TFElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
optimizer = tf.keras.optimizers.Adam(lr=1e-4)

# Load in text strings.
text = "The chef cooked the meal. It was delicious and appetizing, yet I couldn't shake the feeling that Michael Jordan would have the flu game."
# Convert to text tokens
tokens = tokenizer.tokenize(text)  # ['the', 'chef', 'cooked', 'the', 'meal', '.']

# Convert to token ids.
ids = tokenizer.convert_tokens_to_ids(tokens)
ids = np.reshape(ids, (1, len(ids)))  # [1, 6]
ids = tf.constant(ids)


for _ in range(50):
    with tf.GradientTape() as tape:
        # Generate a mask.
        num_masks = 2
        # mask = np.random.choice(np.arange(len(tokens)), size=num_masks, replace=True)
        # Mask should be a boolean array where 1 represents masked token.
        mask_prob = 0.5  # TODO: Change to 0.15
        mask = np.array(np.random.rand(*ids.shape) > 1 - mask_prob, dtype=int)
        tf_mask = tf.constant(mask)

        # Mask the token ids.
        masked_ids = np.where(mask, tokenizer.mask_token_id, ids)
        masked_ids = tf.constant(masked_ids)

        # Convert to tensor.
        inputs = tf.constant(masked_ids)  # [1, 6]

        (adv_logits,) = gen(inputs)  # [1, 6, 30522]
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

        gen_ids = tf_mask * adv_ids + (1 - tf_mask) * ids

        dis_logits = dis(gen_ids)  # [6], logits that
        dis_logits = tf.reshape(dis_logits, [1, -1])  # [1, 6]
        # Linear layer is already in TFElectraDiscriminatorPredictions.
        dis_probs = tf.math.sigmoid(dis_logits)
        dis_preds = tf.cast(dis_probs > 0.5, dtype=tf_mask.dtype)

        # TODO: If generator generates correct token, invert the loss
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

        print(f"Original:            '{tokenizer.decode(ids.numpy().flatten())}'")
        print(f"Masked:              '{tokenizer.decode(masked_ids.numpy().flatten())}'")
        print(f"Generator output:    '{colorize_gen(tokenizer, ids, gen_ids, tf_mask)}'")
        print(f"Discriminator preds: '{colorize_dis(tokenizer, gen_ids, dis_preds)}'")
        print(
            f"gen_loss: {gen_loss:.3f}, dis_loss: {dis_loss:.3f}, gen_acc: {gen_acc:.3f}, dis_acc: {dis_acc:.3f}\n"
        )

    vars = []
    # vars += gen.trainable_variables
    vars += dis.trainable_variables
    grads = tape.gradient(dis_loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    # breakpoint()
