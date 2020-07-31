# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""ResNet, ResNetV2, and ResNeXt models for Keras.

# Reference papers

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
- [Identity Mappings in Deep Residual Networks]
  (https://arxiv.org/abs/1603.05027) (ECCV 2016)
- [Aggregated Residual Transformations for Deep Neural Networks]
  (https://arxiv.org/abs/1611.05431) (CVPR 2017)

# Reference implementations

- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
- [Caffe ResNet]
  (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)
- [Torch ResNetV2]
  (https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua)
- [Torch ResNeXt]
  (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import functools
import numpy as np
from awsdet.models.norms.sync_batch_norm import SyncBatchNormalization

layers = tf.keras.layers
KERNEL_INIT='he_normal'
BN_EPS=1e-5
BN_MOMENTUM=0.9


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, avg_down=False, norm_fn=None,
            name=None, weight_decay=1e-4):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        avg_down: Apply average pooling in the shortcut connection
        norm_fn: BatchNorm or Sync BatchNorm
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    assert norm_fn is not None
    if conv_shortcut is True:
        if not avg_down:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride, use_bias=False, padding='SAME',
                                 kernel_initializer=KERNEL_INIT,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 name=name + '_0_conv')(x)
        else:
            shortcut = layers.AveragePooling2D(pool_size=1, strides=stride, padding='SAME')(x)
            shortcut = layers.Conv2D(4 * filters, 1, strides=1, use_bias=False, padding='SAME',
                                 kernel_initializer=KERNEL_INIT,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 name=name + '_0_conv')(shortcut)
        shortcut = norm_fn(name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False, padding='SAME',
                        kernel_initializer=KERNEL_INIT,
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                        name=name + '_1_conv')(x)
    x = norm_fn(name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME',
                        use_bias=False,
                        kernel_initializer=KERNEL_INIT,
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                        name=name + '_2_conv')(x)

    x = norm_fn(name=name + '_2_bn')(x)

    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, use_bias=False, padding='SAME',
                        kernel_initializer=KERNEL_INIT,
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                        name=name + '_3_conv')(x)

    # https://arxiv.org/pdf/1706.02677.pdf - Last gamma initialized to zeros
    x = norm_fn(gamma_initializer='zeros', name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, avg_down=False, norm_fn=None, name=None, weight_decay=1e-4):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor
        filters: integer, filters of the bottleneck layer in a block
        blocks: integer, blocks in the stacked blocks
        stride1: default 2, stride of the first layer in the first block
        name: string, stack label

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, avg_down=avg_down, norm_fn=norm_fn, name=name + '_block1', weight_decay=weight_decay)
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, avg_down=avg_down, norm_fn=norm_fn, name=name + '_block' + str(i), weight_decay=weight_decay)
    return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, norm_fn=None, name=None, weight_decay=1e-4):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    assert norm_fn is not None
    preact = norm_fn(name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, use_bias=False,
                        padding='SAME', kernel_initializer=KERNEL_INIT,
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                        name=name + '_0_conv')(preact)
    else:
        shortcut = layers.AveragePooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False, padding='SAME',
                        kernel_initializer=KERNEL_INIT,
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                        name=name + '_1_conv')(preact)
    x = norm_fn(name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME', use_bias=False,
                        kernel_initializer=KERNEL_INIT,
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                        name=name + '_2_conv')(x)
    x = norm_fn(name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, use_bias=False, padding='SAME',
                        kernel_initializer=KERNEL_INIT,
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                        name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, norm_fn=None, name=None, weight_decay=1e-4):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, norm_fn=norm_fn, name=name + '_block1', weight_decay=weight_decay) 
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, norm_fn=norm_fn, name=name + '_block' + str(blocks), weight_decay=weight_decay)
    return x


def block3(x, filters, kernel_size=3, stride=1, groups=32, conv_shortcut=True, norm_fn=None, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        groups: default 32, group size for grouped convolution.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    assert norm_fn is not None
    if conv_shortcut is True:
        shortcut = layers.Conv2D((64 // groups) * filters, 1, strides=stride,
                                 use_bias=False, name=name + '_0_conv')(x)
        shortcut = norm_fn(name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = norm_fn(name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c,
                               use_bias=False, name=name + '_2_conv')(x)
    kernel = np.zeros((1, 1, filters * c, filters), dtype=np.float32)
    for i in range(filters):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.
    x = layers.Conv2D(filters, 1, use_bias=False,
                      kernel_initializer={'class_name': 'Constant',
                                          'config': {'value': kernel}},
                      name=name + '_2_gconv')(x)
    x = norm_fn(name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D((64 // groups) * filters, 1,
                      use_bias=False, name=name + '_3_conv')(x)
    x = norm_fn(name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, norm_fn=None, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        groups: default 32, group size for grouped convolution.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups, norm_fn=norm_fn, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False, norm_fn=norm_fn, name=name + '_block' + str(i))
    return x


def ResNet(stack_fn,
           norm_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           weight_decay=1e-4,
           sync_bn_stats=False,
           image_data_format='channels_last',
           deep_stem=False,
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not #FIXME:
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              or the path to the weights file to be loaded.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        weight_decay: L2 weight decay multiplier passed in to all layers (excluding biases, beta and gamma)
        sync_bn_stats: whether to sync BN stats across all workers in distributed training setup

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    input_shape = (None, None, 3) 
    img_input = layers.Input(shape=input_shape)
    x = img_input

    if not deep_stem:
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
        x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv', 
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                kernel_initializer=KERNEL_INIT)(x)
    else: # replace 7x7 with 3 3x3 convs to get same receptive field (resnetv1_c)
        x = layers.Conv2D(64, 3, strides=2, padding='SAME', use_bias=False, name='conv1_3x3_stem1',
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                kernel_initializer=KERNEL_INIT)(x)
        x = norm_fn(name='conv1_3x3_1_bn')(x)
        x = layers.Activation('relu', name='conv1_3x3_1_relu')(x)

        x = layers.Conv2D(64, 3, strides=1, padding='SAME', use_bias=False, name='conv1_3x3_stem2',
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                kernel_initializer=KERNEL_INIT)(x)
        x = norm_fn(name='conv1_3x3_2_bn')(x)
        x = layers.Activation('relu', name='conv1_3x3_2_relu')(x)
        x = layers.Conv2D(128, 3, strides=1, padding='SAME', use_bias=False, name='conv1_3x3_stem3',
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                kernel_initializer=KERNEL_INIT)(x)

    if preact is False:
        x = norm_fn(name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.MaxPooling2D(3, strides=2, padding='SAME', name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True: # resnetv2
        x = norm_fn(name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='logits')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=model_name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


def ResNet50V1_b(include_top=True,
             weights=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             weight_decay=1e-4,
             norm_fn=None,
             **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack1(x, 128, 4, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack1(x, 256, 6, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack1(x, 512, 3, norm_fn=norm_fn, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, False, True, 'resnet50v1_b',
                  include_top, weights, input_shape, pooling, classes, weight_decay, **kwargs)


def ResNet50V1_c(include_top=True,
             weights=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             weight_decay=1e-4,
             norm_fn=None,
             **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack1(x, 128, 4, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack1(x, 256, 6, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack1(x, 512, 3, norm_fn=norm_fn, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, False, True, 'resnet50v1_c',
                  include_top, weights, input_shape, pooling, classes, weight_decay,
                  deep_stem=True, **kwargs)


def ResNet50V1_d(include_top=True,
             weights=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             weight_decay=1e-4,
             norm_fn=None,
             **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, norm_fn=norm_fn, stride1=1, avg_down=True, name='conv2', weight_decay=weight_decay)
        x = stack1(x, 128, 4, norm_fn=norm_fn, name='conv3', avg_down=True, weight_decay=weight_decay)
        x = stack1(x, 256, 6, norm_fn=norm_fn, name='conv4', avg_down=True, weight_decay=weight_decay)
        x = stack1(x, 512, 3, norm_fn=norm_fn, name='conv5', avg_down=True, weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, False, True, 'resnet50v1_d',
                  include_top, weights, input_shape, pooling, classes, weight_decay,
                  deep_stem=True, **kwargs)


def ResNet101V1_b(include_top=True,
              weights=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              weight_decay=1e-4,
              norm_fn=None,
              **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack1(x, 128, 4, name='conv3', norm_fn=norm_fn, weight_decay=weight_decay)
        x = stack1(x, 256, 23, name='conv4', norm_fn=norm_fn, weight_decay=weight_decay)
        x = stack1(x, 512, 3, name='conv5', norm_fn=norm_fn, weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, False, True, 'resnet101v1_b',
                  include_top, weights, input_shape, pooling, classes, **kwargs)


def ResNet101V1_c(include_top=True,
              weights=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              weight_decay=1e-4,
              sync_bn_stats=False,
              **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack1(x, 128, 4, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack1(x, 256, 23, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack1(x, 512, 3, norm_fn=norm_fn, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, False, True, 'resnet101v1_c',
                  include_top, weights, input_shape, pooling, classes,
                  deep_stem=True, **kwargs)


def ResNet101V1_d(include_top=True,
              weights=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              weight_decay=1e-4,
              norm_fn=None,
              **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, avg_down=True, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack1(x, 128, 4, avg_down=True, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack1(x, 256, 23, avg_down=True, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack1(x, 512, 3, avg_down=True, norm_fn=norm_fn, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, False, True, 'resnet101v1_d',
                  include_top, weights, input_shape, pooling, classes, deep_stem=True, **kwargs)


def ResNet152(include_top=True,
              weights=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              weight_decay=1e-4,
              norm_fn=None,
              **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack1(x, 128, 8, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack1(x, 256, 36, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack1(x, 512, 3, norm_fn=norm_fn, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, False, True, 'resnet152',
                  include_top, weights, input_shape, pooling, classes, **kwargs) 


def ResNet50V2(include_top=True,
               weights=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               weight_decay=1e-4,
               norm_fn=None,
               **kwargs):
    def stack_fn(x):
        x = stack2(x, 64, 3, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack2(x, 128, 4, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack2(x, 256, 6, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack2(x, 512, 3, norm_fn=norm_fn, stride1=1, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, True, True, 'resnet50v2',
                  include_top, weights, input_shape, pooling, classes, weight_decay, **kwargs)


def ResNet101V2(include_top=True,
                weights=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                weight_decay=1e-4,
                norm_fn=None,
                **kwargs):
    def stack_fn(x):
        x = stack2(x, 64, 3, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack2(x, 128, 4, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack2(x, 256, 23, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack2(x, 512, 3, norm_fn=norm_fn, stride1=1, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, True, True, 'resnet101v2',
                  include_top, weights, input_shape, pooling, classes, **kwargs)


def ResNet152V2(include_top=True,
                weights=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                weight_decay=1e-4,
                norm_fn=None,
                **kwargs):
    def stack_fn(x):
        x = stack2(x, 64, 3, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack2(x, 128, 8, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack2(x, 256, 36, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack2(x, 512, 3, stride1=1, norm_fn=norm_fn, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, True, True, 'resnet152v2',
                  include_top, weights, input_shape, pooling, classes, **kwargs)


def ResNeXt50(include_top=True,
              weights=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              weight_decay=1e-4,
              norm_fn=None,
              **kwargs):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack3(x, 256, 4, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack3(x, 512, 6, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack3(x, 1024, 3, norm_fn=norm_fn, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, False, False, 'resnext50',
                  include_top, weights, input_shape, pooling, classes, **kwargs)


def ResNeXt101(include_top=True,
               weights=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               weight_decay=1e-4,
               norm_fn=None,
               **kwargs):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, norm_fn=norm_fn, name='conv2', weight_decay=weight_decay)
        x = stack3(x, 256, 4, norm_fn=norm_fn, name='conv3', weight_decay=weight_decay)
        x = stack3(x, 512, 23, norm_fn=norm_fn, name='conv4', weight_decay=weight_decay)
        x = stack3(x, 1024, 3, norm_fn=norm_fn, name='conv5', weight_decay=weight_decay)
        return x
    return ResNet(stack_fn, norm_fn, False, False, 'resnext101',
                  include_top, weights, input_shape, pooling, classes, **kwargs)


def build_resnet(model_name,
                    include_top=True,
                    weights=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000,
                    weight_decay=1e-4,
                    image_data_format='channels_last',
                    sync_bn=False,
                    **kwargs):
    
    bn_axis = 3 if image_data_format == 'channels_last' else 1
    norm_fn = None
    if not sync_bn:
        norm_fn = functools.partial(layers.BatchNormalization, axis=bn_axis, epsilon=BN_EPS, momentum=BN_MOMENTUM)
    else:
        norm_fn = functools.partial(SyncBatchNormalization, axis=bn_axis, epsilon=BN_EPS, momentum=BN_MOMENTUM)

    if model_name == 'ResNet50V1_b':
        return ResNet50V1_b(include_top=include_top,
                norm_fn=norm_fn,
                weights=weights, 
                input_shape=input_shape, 
                pooling=pooling,
                classes=classes,
                weight_decay=weight_decay)
    elif model_name == 'ResNet50V1_d':
        return ResNet50V1_d(include_top=include_top,
                norm_fn=norm_fn,
                weights=weights, 
                input_shape=input_shape, 
                pooling=pooling,
                classes=classes,
                weight_decay=weight_decay)
    elif model_name == 'ResNet101V1_b':
        return ResNet101V1_b(include_top=include_top,
                norm_fn=norm_fn,
                weights=weights, 
                input_shape=input_shape, 
                pooling=pooling,
                classes=classes,
                weight_decay=weight_decay)
    elif model_name == 'ResNet101V1_d':
        return ResNet101V1_d(include_top=include_top,
                norm_fn=norm_fn,
                weights=weights, 
                input_shape=input_shape, 
                pooling=pooling,
                classes=classes,
                weight_decay=weight_decay)
    else:
        raise NotImplementedError



