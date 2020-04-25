# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from awsdet.models.utils.misc import calc_img_shapes, calc_batch_padded_shape


class AnchorGenerator:
    """
    This class operates on a padded image, eg. [1216, 1216]
    and generate scales*ratios number of anchor boxes for each point in
    padded image, with stride = feature_strides
    number of anchor = (1216 // feature_stride)^2
    number of anchor boxes = number of anchor * (scales_len*ratio_len)
    """
    def __init__(self,
                 scales=(32, 64, 128, 256, 512),
                 ratios=(0.5, 1, 2),
                 feature_strides=(4, 8, 16, 32, 64),
                 padding_size=None):
        '''
        Anchor Generator
        
        Attributes
        ---
            scales: 1D array of anchor sizes in pixels.
            ratios: 1D array of anchor ratios of width/height.
            feature_strides: Stride of the feature map relative to the image in pixels.
        '''
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides
        self.anchors = tf.stop_gradient(self.generate_anchors(padding_size))
        self.padding_size = padding_size


    def generate_anchors(self, pad_shape):
        feature_shapes = [(tf.math.ceil(pad_shape[0] / stride),
                           tf.math.ceil(pad_shape[1] / stride))
                          for stride in self.feature_strides]

        anchors = [
            self._generate_level_anchors(level, feature_shape)
            for level, feature_shape in enumerate(feature_shapes)
        ]
        
        anchors = tf.concat(anchors, axis=0)
        return anchors


    @tf.function(experimental_relax_shapes=True)
    def generate_pyramid_anchors(self, img_metas):
        '''
        Generate the multi-level anchors for Region Proposal Network
        
        Args
        ---
            img_metas: [batch_size, 11]
        
        Returns
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
        '''
        
        # generate valid flags
        img_shapes = calc_img_shapes(img_metas) # (800, 1067)
        batch_size = tf.shape(img_shapes)[0]
        valid_flags = tf.TensorArray(tf.int32, size=batch_size)

        for i in range(batch_size):
            valid_flags = valid_flags.write(i, self._generate_valid_flags(self.anchors, img_shapes[i]))
        valid_flags = valid_flags.stack()
        
        valid_flags = tf.stop_gradient(valid_flags)
        
        return self.anchors, valid_flags


    @tf.function(experimental_relax_shapes=True)
    def _generate_valid_flags(self, anchors, img_shape):
        '''
        Find valid anchors. Mark anchor boxes on padded area as invalid.
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            img_shape: Tuple. (height, width, channels)
            
        Returns
        ---
            valid_flags: [num_anchors]
        '''
        H = img_shape[0] # tf.cast(img_shape[0], tf.float32)
        W = img_shape[1] # tf.cast(img_shape[1], tf.float32)

        y_center = (anchors[:, 2] + anchors[:, 0]) / 2
        x_center = (anchors[:, 3] + anchors[:, 1]) / 2

        valid_flags = tf.ones(anchors.shape[0], dtype=tf.int32)
        zeros = tf.zeros(anchors.shape[0], dtype=tf.int32)
        # set boxes whose center is out of image area as invalid.
        valid_flags = tf.where(y_center >= tf.constant(0.0, dtype=H.dtype), valid_flags, zeros)
        valid_flags = tf.where(x_center >= tf.constant(0.0, dtype=W.dtype), valid_flags, zeros)
        valid_flags = tf.where(y_center <= H, valid_flags, zeros)
        valid_flags = tf.where(x_center <= W, valid_flags, zeros)
        return valid_flags

    def _generate_level_anchors(self, level, feature_shape):
        '''
        Generate the anchors given the spatial shape of feature map.
        
        Returns: 
            [anchors_num, (y1, x1, y2, x2)]
        '''
        scale = self.scales[level]
        ratios = self.ratios
        feature_stride = self.feature_strides[level]

        # Get all combinations of scales and ratios
        scales, ratios = tf.meshgrid([float(scale)], ratios)

        scales = tf.reshape(scales, [-1])
        ratios = tf.reshape(ratios, [-1])

        # Enumerate heights and widths from scales and ratios
        heights = scales / tf.sqrt(ratios)
        widths = scales * tf.sqrt(ratios)

        # Enumerate shifts in feature space, [0, 4, ..., 1216-4]
        shifts_y = tf.multiply(tf.range(feature_shape[0]), feature_stride)
        shifts_x = tf.multiply(tf.range(feature_shape[1]), feature_stride)

        shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(
            shifts_y, tf.float32)
        shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = tf.reshape(
            tf.stack([box_centers_y, box_centers_x], axis=2), (-1, 2))
        box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2),
                               (-1, 2))
        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = tf.concat(
            [box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes],
            axis=1)
        return boxes
