# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras


@tf.function(experimental_relax_shapes=True)
def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
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


def focal_loss(y_preds, y_true, alpha=0.25, gamma=2.0, avg_factor=1.0, num_classes=80):
    """
    Args:
        y_preds: [batch size * num_anchors, num_classes]
        y_true: [batch size * num_anchors, ]
        alpha: class balance factor (binary classification)
        gamma: focus strength
        avg_factor: value to divide the loss sum by (equals num_pos_anchors for RetinaNet)
        num_classes:
    Return:
        Scalar loss normalized by number of anchors that got assigned to GT (TODO: generalize this take normalization factor as an argument)
    """
    assert gamma >= 0.0
    # print('target min max', tf.reduce_min(y_true), tf.reduce_max(y_true))
    pred_sigmoid = tf.nn.sigmoid(y_preds)
    oh_target = tf.one_hot(y_true-1, depth=num_classes)
    positive_mask = tf.math.equal(oh_target, 1)
    avg_factor = tf.math.maximum(1.0, tf.cast(avg_factor, tf.float32))
    # print(tf.shape(oh_target), tf.shape(y_preds))
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=oh_target, logits=y_preds)
    # print('ce', ce)
    pt = tf.where(positive_mask, pred_sigmoid, 1.0 - pred_sigmoid)
    loss = tf.math.pow(1.0 - pt, gamma) * ce
    weighted_loss = tf.where(positive_mask, alpha * loss, (1.0 - alpha) * loss)
    # print('loss', weighted_loss)
    batch_loss_sum = tf.reduce_sum(weighted_loss)
    # print('batch loss sum', batch_loss_sum, avg_factor)
    return batch_loss_sum / avg_factor


def retinanet_bbox_loss(deltas, target_deltas, avg_factor=1.0):
    '''Return the Retinanet bounding box loss    
    Args
    ---
        deltas: [batch * anchors, (dy, dx, log(dh), log(dw))]
        target_deltas: [batch * anchors, (dy, dx, log(dh), log(dw))]
        inside_weights: [batch * anchors, 4] weights for inside targets
        outside_weights: [batch * anchors, 4] weights for outside targets        
    '''
    #print('bbox loss', tf.shape(deltas), tf.shape(target_deltas), avg_factor)
    loss = tf.math.abs(deltas - target_deltas)
    avg_factor = tf.math.maximum(1.0, tf.cast(avg_factor, tf.float32))
    return tf.reduce_sum(loss) / avg_factor
    return loss


def rpn_class_loss(logits, labels, avg_factor=256.0, weight=1.0, label_smoothing=0.0):
    """
    :param weight:
    :param logits: [batch size * num_anchors, 2]
    :param labels: [batch size * num_anchors, ]
    :return:
    """
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=2)
    batch_loss_sum = tf.reduce_sum(
                        tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits))
    return batch_loss_sum / avg_factor


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


def rcnn_class_loss(logits, labels, avg_factor=512.0, weight=1.0, label_smoothing=0.0):
    """
    :param weight:
    :param logits: [batch size * num_anchors, 2]
    :param labels: [batch size * num_anchors, ]
    :return:
    """
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=81)
    batch_loss_sum = tf.reduce_sum(
                        tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits))
    return batch_loss_sum / avg_factor


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

