# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
from collections import defaultdict
import pycocotools.mask as mask_util

def mask2result(masks, labels, meta, num_classes=81, threshold=0.5):
    meta = np.squeeze(meta)
    img_heights, img_widths = meta[:2].astype(np.int32)
    unpadded_height = tf.cast(meta[3], tf.int32)
    unpadded_width = tf.cast(meta[4], tf.int32)
    orig_height = tf.cast(meta[0], tf.int32)
    orig_width = tf.cast(meta[1], tf.int32)
    masks = masks[:,:unpadded_height,:unpadded_width, :]
    masks = tf.image.resize(masks, (orig_height, orig_width), method='nearest')
    masks_np = np.squeeze((masks.numpy()>threshold).astype(np.int32))
    labels_np = labels.numpy()
    if meta[-1]==1:
        masks_np = np.flip(masks_np, axis=2)
    lists = defaultdict(list)
    for i,j in enumerate(labels_np):
        lists[j].append(mask_util.encode(
                    np.array(
                        masks_np[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])
    return lists
