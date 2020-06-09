"""
What's the best way to develop a new pretraining script?

Dynamic masking straight from text.
Abtract out the gradient accumulation functionality. Tracking loss, acc variables within the accumulator rather than outside.
Incorporate the new transformers version. Be willing to lose my current work.

# TODO: Should we include special tokens? <BOS>, <EOS>.

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

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def tf_tensor(arr: np.array) -> tf.Tensor:
    return tf.expand_dims(tf.constant(arr), 0)


# TODO: Should I use bert-base-uncased?
tokenizer = ElectraTokenizer.from_pretrained("bert-base-uncased")

gen_config = ElectraConfig.from_pretrained("google/electra-small-generator")
dis_config = ElectraConfig.from_pretrained("google/electra-small-discriminator")

gen = TFElectraForMaskedLM.from_pretrained("google/electra-small-generator")
dis = TFElectraForPreTraining.from_pretrained("google/electra-small-discriminator")

# Load in text strings.
text = "The chef cooked the meal."
# Convert to text tokens
tokens = tokenizer.tokenize(text)  # ['the', 'chef', 'cooked', 'the', 'meal', '.']

# Convert to token ids.
ids = tokenizer.convert_tokens_to_ids(tokens)

# Generate a mask.
num_masks = 2
# mask = np.random.choice(np.arange(len(tokens)), size=num_masks, replace=True)
# Mask should be a boolean array where 1 represents masked token.
mask_prob = 0.5  # TODO: Change to 0.15
mask = np.array(np.random.rand(len(tokens)) > 1 - mask_prob, dtype=int)
tf_mask = tf_tensor(mask)

# Mask the token ids.
masked_ids = np.where(mask, tokenizer.mask_token_id, ids)

# Convert to tensor.
inputs = tf_tensor(masked_ids)  # [1, 6]

(adv_probs,) = gen(inputs)  # [1, 6, 30522]
# gen_loss = E[-log(x_i)] for i in mask_locs

adv_ids = tf.argmax(adv_probs, axis=-1)  # [1, 6]

dis_logits = dis(adv_ids)  # [6], logits that
dis_logits = tf.reshape(dis_logits, [1, -1])  # [1, 6]
# TODO: Add a linear layer here?
dis_probs = tf.math.sigmoid(dis_logits)

# TODO: If generator generates correct token, invert the loss
bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)  # [b]
loss = bce(y_true=tf_mask, y_pred=dis_logits)


breakpoint()
