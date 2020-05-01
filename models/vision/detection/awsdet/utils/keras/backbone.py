# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
import awsdet

def get_base_model(model_name, weights_path, weight_decay=1e-4):
    """
        Define base model used in transfer learning.
    """
    if not weights_path:
        weights_path = 'imagenet'
    if model_name == 'VGG16':
        base_model = VGG16(weights=weights_path, include_top=False)
    elif model_name == 'VGG19':
        base_model = VGG19(weights=weights_path, include_top=False)
    elif model_name == 'ResNet50V1':
        base_model = awsdet.models.backbones.ResNet50(weights=None, include_top=False, weight_decay=weight_decay)
    elif model_name == 'ResNet50V2':
        base_model = awsdet.models.backbones.ResNet50V2(weights=None, include_top=False, weight_decay=weight_decay)
    elif model_name == 'Xception':
        base_model = Xception(weights=weights_path, include_top=False)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights=weights_path, include_top=False)
    elif model_name == 'InceptionResNetV2':
        base_model = InceptionResNetV2(weights=weights_path,
                                        include_top=False)
    elif model_name == 'MobileNet':
        base_model = MobileNet(weights=weights_path, include_top=False)
    else:
        raise ValueError(
            'Valid base model values are: "VGG16","VGG19","ResNet50V1", "ResNet50V2", "Xception", \
                            "InceptionV3","InceptionResNetV2","MobileNet".'
        )
    return base_model


def get_outputs(model):
    if model.name == "resnet50":
        return [model.get_layer(l).output for l in ['conv2_block3_out',
            'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']]
    elif model.name == "resnet50v2":
        return [model.get_layer(l).output for l in ['conv2_block3_out',
            'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']]
    raise NotImplementedError


if __name__ == "__main__":

    m = get_base_model("ResNet50")
    print("Input:", m.input)
    print("Outputs:")
    for idx, l in enumerate(m.layers):
        print(idx, l.name, l.output)
    model = tf.keras.Model(
        inputs=m.input,
        outputs=[m.layers[i].output for i in [38, 80, 142, 174]])
    m.summary()
    print([o.name for o in get_outputs(m)])
