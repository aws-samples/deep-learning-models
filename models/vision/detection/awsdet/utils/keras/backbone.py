# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
import awsdet


def get_base_model(model_name, weights_path, weight_decay=1e-4):
    """
        Define base model used in transfer learning.
    """
    if not weights_path:
        weights_path = 'imagenet'
    if model_name == 'ResNet50V1':
        base_model = awsdet.models.backbones.ResNet50(weights=None, include_top=False, weight_decay=weight_decay)
    elif model_name == 'ResNet50V1_AWS':
        base_model = awsdet.models.backbones.ResNet50V1(weights=None, include_top=False, weight_decay=weight_decay)
    elif model_name == 'ResNet50V2':
        base_model = awsdet.models.backbones.ResNet50V2(weights=None, include_top=False, weight_decay=weight_decay)
    elif model_name == "ResNet101V1":
        base_model = awsdet.models.backbones.ResNet101(weights=None, include_top=False, weight_decay=weight_decay)
    elif model_name == 'HRNetV2p':
        base_model = awsdet.models.backbones.build_hrnet('hrnet_w32c', include_top=False)
    else:
        raise ValueError(
            'Valid base model values are: "ResNet50V1", "ResNet50V2", "ResNet101V1, "HRNetV2p"'
        )
    return base_model
 

def get_outputs(model):
    if model.name in ['resnet50', 'resnet50v2']:
        return [model.get_layer(l).output for l in ['conv2_block3_out',
            'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']]
    elif model.name == 'resnet101':
        return [model.get_layer(l).output for l in ['conv2_block3_out',
            'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out']]
    elif model.name == 'hr_net':
        stage4_outputs = model.get_layer('s4').output
        return stage4_outputs

    raise NotImplementedError


if __name__ == "__main__":
    m = get_base_model("ResNet101V1", "/workspace/shared_workspace/models/vision/detection/weights/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5")
    print("Input:", m.input)
    print("Outputs:")
    for idx, l in enumerate(m.layers):
        print(idx, l.name, l.output)
    model = tf.keras.Model(
        inputs=m.input,
        outputs=[m.layers[i].output for i in [38, 80, 142, 174]])
    m.summary()
    print([o.name for o in get_outputs(m)])
 
