# Implementation of Darknet-53 to train Yolo-v3

import tensorflow as tf 

def conv_unit(x, filters, kernel, strides, weight_decay=1e-4, last=False):
    """ A single convolutional unit that consists of 
        Conv2D layer, BatchNorm, and LeakyReLU activation.
    """

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel,
            use_bias=False,kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               strides=strides, padding='same')(x)
    if not last:
        x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.9)(x)
    else:
        x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.9,
                gamma_initializer='zeros')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(inputs, filters):
    """ A single residual block built from conv_unit layers """

    x = conv_unit(inputs, filters=filters, kernel=(1, 1), strides=1)
    x = conv_unit(x, filters=2 * filters, kernel=(3, 3), strides=1, last=True)
    x = tf.keras.layers.add([inputs, x])
    return x

def stack(inputs, filters, n):
    """ Stacks n residual blocks on top of each other """

    x = residual_block(inputs, filters=filters)
    for _ in range(n-1):
        x = residual_block(x, filters=filters)
    return x

def darknet_base(inputs):
    """ The base of Darknet 53 without the head """

    x = conv_unit(inputs, filters=32, kernel=(3, 3), strides=1)
    x = conv_unit(inputs, filters=64, kernel=(3, 3), strides=2)

    x = stack(x, filters=32, n=1)
    x = conv_unit(x, filters=128, kernel=(3, 3), strides=2)
    
    x = stack(x, filters=64, n=2)
    x = conv_unit(x, filters=256, kernel=(3, 3), strides=2)

    x = stack(x, filters=128, n=8)
    x = conv_unit(x, filters=512, kernel=(3, 3), strides=2)

    x = stack(x, filters=256, n=8)
    x = conv_unit(x, filters=1024, kernel=(3, 3), strides=2)

    x = stack(x, filters=512, n=4)

    return x


def Darknet(model_name='darknet53', weight_decay=1e-4,
            include_top=True):
    """ Instantiates the Darknet-53 architecture
        
        # Arguments
            model_name: Name of the model
            include_top: Whether to include the head of the model.
                True for classification, False for detection.
    """
    inputs = tf.keras.layers.Input(shape=(None, None, 3))
    x = darknet_base(inputs)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1000,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        # x = tf.keras.layers.Activation('softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

