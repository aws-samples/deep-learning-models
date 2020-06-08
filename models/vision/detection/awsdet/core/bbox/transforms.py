# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from ...models.utils import (parse_image_meta)

def bbox2delta(box, gt_box, target_means, target_stds):
    '''Compute refinement needed to transform box to gt_box.
    
    Args
    ---
        box: [..., (y1, x1, y2, x2)]
        gt_box: [..., (y1, x1, y2, x2)]
        target_means: [4]
        target_stds: [4]
    '''
    target_means = tf.constant(target_means, dtype=tf.float32)
    target_stds = tf.constant(target_stds, dtype=tf.float32)

    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[..., 2] - box[..., 0] 
    width = box[..., 3] - box[..., 1]
    center_y = (box[..., 0] + box[..., 2]) * 0.5
    center_x = (box[..., 1] + box[..., 3]) * 0.5

    gt_height = gt_box[..., 2] - gt_box[..., 0] 
    gt_width = gt_box[..., 3] - gt_box[..., 1] 
    gt_center_y = (gt_box[..., 0] + gt_box[..., 2]) * 0.5
    gt_center_x = (gt_box[..., 1] +  gt_box[..., 3]) * 0.5

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    delta = tf.stack([dy, dx, dh, dw], axis=-1)
    delta = (delta - target_means) / target_stds

    return delta


def delta2bbox(box, delta, target_means, target_stds):
    '''Compute bounding box based on roi and delta.
   
    Args
    ---
        box: [N, (y1, x1, y2, x2)] box to update
        delta: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        target_means: [4]
        target_stds: [4]
    '''
    target_means = tf.constant(target_means, dtype=tf.float32)
    target_stds = tf.constant(target_stds, dtype=tf.float32)
    denorm_delta = delta * target_stds + target_means
    dy, dx, dh, dw = denorm_delta[:,0], denorm_delta[:,1], denorm_delta[:,2], denorm_delta[:,3] 
    max_ratio = np.abs(np.log(16.0/1000)) #TODO: make part of config
    dw = tf.clip_by_value(dw, clip_value_min=-max_ratio, clip_value_max=max_ratio)
    dh = tf.clip_by_value(dh, clip_value_min=-max_ratio, clip_value_max=max_ratio)
    # Convert to y, x, h, w
    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width
   
    # Apply delta
    center_y += dy * height
    center_x += dx * width
    height *= tf.exp(dh)
    width *= tf.exp(dw)
   
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1)
    return result


def bbox_clip(box, window):
    '''
    Args
    ---
        box: [N, (y1, x1, y2, x2)]
        window: [4] in the form y1, x1, y2, x2
    '''
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(box, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1)
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


def bbox_flip(bboxes, width):
    '''
    Flip bboxes horizontally.
    
    Args
    ---
        bboxes: [..., 4]
        width: Int or Float
    '''
    y1, x1, y2, x2 = tf.split(bboxes, 4, axis=-1)

    new_x1 = width - x2
    new_x2 = width - x1

    flipped = tf.concat([y1, new_x1, y2, new_x2], axis=-1)

    return flipped


def bbox_mapping(box, img_meta):
    '''
    Map bboxes from the original image scale to testing scale
    Args
    ---
        box: [N, 4]
        img_meta: [11]
    '''
    img_meta = parse_image_meta(img_meta)
    scale = img_meta['scale']
    flip = img_meta['flip']

    box = box * scale
    if tf.equal(flip, 1):
        box = bbox_flip(box, img_meta['img_shape'][1])

    return box


def bbox_mapping_back(box, img_meta):
    '''
    Map bboxes from testing scale to original image scale
    Args
    ---
        box: [N, 4]
        img_meta: [11]
    '''
    img_meta = parse_image_meta(img_meta)
    scale = img_meta['scale']
    flip = img_meta['flip']
    if tf.equal(flip, 1):
        box = bbox_flip(box, img_meta['img_shape'][1])
    box = box / scale

    return box


def bbox2result(bboxes, labels, scores, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 4)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """

    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = tf.concat([bboxes,tf.expand_dims(scores, axis=1)], axis=-1)
        bboxes = bboxes.numpy()
        labels = labels.numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]

