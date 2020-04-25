# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras


@tf.function(experimental_relax_shapes=True)
def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1], scale_factor=2**16):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    sign = tf.stop_gradient(tf.cast(tf.less(abs_in_box_diff, 1. / sigma_2), tf.float32))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_sum(out_loss_box, axis=dim) # outside weight gives a fixed denominator
    loss_box = tf.cast(loss_box, tf.float32)
    return loss_box


def rpn_class_loss(logits, labels, rpn_deltas=256.0, weight=1.0, label_smoothing=0.0):
    """
    :param weight:
    :param logits: [batch size * num_anchors, 2]
    :param labels: [batch size * num_anchors, ]
    :return:
    """
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=2)
    batch_loss_sum = tf.reduce_sum(tf.compat.v1.losses.softmax_cross_entropy(
            logits=logits,
            onehot_labels=onehot_labels, label_smoothing=label_smoothing,
            weights=weight, reduction=tf.compat.v1.losses.Reduction.NONE)) / rpn_deltas # 256 targets per example
    return batch_loss_sum


def rpn_bbox_loss(rpn_deltas, target_deltas, rpn_inside_weights, rpn_outside_weights):
    '''Return the RPN bounding box loss    
    Args
    ---
        rpn_deltas: [batch * anchors, (dy, dx, log(dh), log(dw))]
        target_deltas: [batch * anchors, (dy, dx, log(dh), log(dw))]
        rpn_inside_weights: [batch * anchors, 4] weights for inside targets
        rpn_outside_weights: [batch * anchors, 4] weights for outside targets        
    '''
    loss = smooth_l1_loss(rpn_deltas,
                              target_deltas,
                              rpn_inside_weights,
                              rpn_outside_weights,
                              sigma=3.0, dim=[0, 1])
    return loss


def rcnn_class_loss(logits, labels, roi_deltas=512.0, weight=1.0, label_smoothing=0.0):
    """
    :param weight:
    :param logits: [batch size * num_anchors, 2]
    :param labels: [batch size * num_anchors, ]
    :return:
    """
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=81)
    batch_loss_sum = tf.reduce_sum(tf.compat.v1.losses.softmax_cross_entropy(
            logits=logits,
            onehot_labels=onehot_labels, label_smoothing=label_smoothing,
            weights=weight, reduction=tf.compat.v1.losses.Reduction.NONE)) / roi_deltas # 512 targets per example
    return batch_loss_sum

def rcnn_bbox_loss(roi_deltas, target_deltas, roi_inside_weights, roi_outside_weights):
    '''Return the RCNN ROI box loss    
    Args
    ---
        roi_deltas: [batch * anchors, (dy, dx, log(dh), log(dw))]
        target_deltas: [batch * anchors, (dy, dx, log(dh), log(dw))]
        roi_inside_weights: [batch * anchors, 4] weights for inside targets
        roi_outside_weights: [batch * anchors, 4] weights for outside targets        
    '''
    loss = smooth_l1_loss(roi_deltas,
                              target_deltas,
                              roi_inside_weights,
                              roi_outside_weights,
                              sigma=1.0,
                              dim=[0, 1])
    return loss

