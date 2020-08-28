import tensorflow as tf
from preprocessing.augmentation_utils import mixup
import horovod.tensorflow as hvd

layers = tf.keras.layers

@tf.function
def train_step(model, opt, loss_func, images, labels, first_batch, batch_size, mixup_alpha=0.0, fp32=False):
    images, labels = mixup(batch_size, mixup_alpha, images, labels)
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss_func(labels, tf.cast(logits, tf.float32))
        loss_value += tf.add_n(model.losses)
        if not fp32:
            scaled_loss_value = opt.get_scaled_loss(loss_value)

    tape = hvd.DistributedGradientTape(tape, compression=hvd.Compression.fp16)
    if not fp32:
        grads = tape.gradient(scaled_loss_value, model.trainable_variables)
        grads = opt.get_unscaled_gradients(grads)
    else:
        grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    
    probs = layers.Activation('softmax', dtype='float32')(logits)
    top_1_pred = tf.squeeze(tf.math.top_k(probs, k=1)[1])
    sparse_labels = tf.cast(tf.math.argmax(labels, axis=1), tf.int32)
    top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, sparse_labels), tf.int32))
    return loss_value, top_1_accuracy


@tf.function
def validation_step(images, labels, model, loss_func):
    pred = model(images, training=False)
    loss = loss_func(labels, pred)
    top_1_pred = tf.squeeze(tf.math.top_k(pred, k=1)[1])
    sparse_labels = tf.cast(tf.math.argmax(labels, axis=1), tf.int32)
    top_1_accuracy = tf.math.reduce_sum(tf.cast(tf.equal(top_1_pred, sparse_labels), tf.int32))
    return loss, top_1_accuracy


