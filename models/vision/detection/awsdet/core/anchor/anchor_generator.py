# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from awsdet.models.utils.misc import calc_img_shapes, calc_pad_shapes


class AnchorGenerator:
    """
    Standard anchor generator for 2D anchor-based detectors

    Args:
        strides (list[int]): Strides of anchors in multiple feature levels.
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes of anchors in multiple
            levels. If None is given, strides will be used as base_sizes.
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales.
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
    """
    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = strides
        self.base_sizes = list(strides) if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = tf.constant(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = [2**(i / scales_per_octave) for i in range(scales_per_octave)]
            scales = [octave_base_scale * octave_scale for octave_scale in octave_scales] 
            self.scales = tf.constant(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = tf.constant(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

       
    @property
    def num_base_anchors(self):
        return [tf.shape(base_anchors)[0] for base_anchors in self.base_anchors]


    @property
    def num_levels(self):
        return len(self.strides)

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors


    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):

        w = base_size
        h = base_size

        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = tf.sqrt(ratios)
        w_ratios = 1 / h_ratios

        if self.scale_major:
            ws = tf.reshape((w * w_ratios[:, None] * scales[None, :]), [-1])
            hs = tf.reshape((h * h_ratios[:, None] * scales[None, :]), [-1])
        else:
            ws = tf.reshape((w * scales[:, None] * w_ratios[None, :]), [-1])
            hs = tf.reshape((h * scales[:, None] * h_ratios[None, :]), [-1])

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = tf.stack(base_anchors, axis=-1)
        return base_anchors


    def grid_anchors(self, featmap_sizes):
        """Generate grid anchors in multiple feature levels
        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
        Return:
            list[Tensor]: Anchors in multiple feature levels.
                The sizes of each tensor should be [N, 4], where
                N = width * height * num_base_anchors, width and height
                are the sizes of the corresponding feature level,
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i],
                featmap_sizes[i],
                self.strides[i])
            multi_level_anchors.append(anchors)
        return multi_level_anchors


    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=16):
        feat_h = featmap_size[0]
        feat_w = featmap_size[1]
        shifts_x = tf.multiply(tf.range(feat_w), stride)
        shifts_y = tf.multiply(tf.range(feat_h), stride)
        shift_xx, shift_yy = tf.meshgrid(shifts_x, shifts_y)
        shift_xx = tf.reshape(shift_xx, [-1])
        shift_yy = tf.reshape(shift_yy, [-1])
        shifts = tf.stack([shift_yy, shift_xx, shift_yy, shift_xx], axis=-1)
        shifts = tf.cast(shifts, base_anchors.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = tf.reshape(all_anchors, (-1, 4))
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return tf.stop_gradient(all_anchors)


    def valid_flags(self, featmap_sizes, pad_shape):
        """Generate valid flags of anchors in multiple feature levels
        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
        Return:
            list(Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[0], pad_shape[1]
            valid_feat_h = tf.math.minimum(tf.cast(tf.math.ceil(h / anchor_stride), tf.int32), feat_h)
            valid_feat_w = tf.math.minimum(tf.cast(tf.math.ceil(w / anchor_stride), tf.int32), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i])
            multi_level_flags.append(flags)
        return multi_level_flags


    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        valid_x = tf.zeros(feat_w, dtype=tf.int32)
        valid_y = tf.zeros(feat_h, dtype=tf.int32)
        indices_w = tf.expand_dims(tf.range(valid_w), axis=1)
        indices_h = tf.expand_dims(tf.range(valid_h), axis=1)
        valid_x = tf.tensor_scatter_nd_update(valid_x, indices_w, tf.ones(valid_w, dtype=tf.int32))
        valid_y = tf.tensor_scatter_nd_update(valid_y, indices_h, tf.ones(valid_h, dtype=tf.int32))
        grid = tf.meshgrid(valid_y, valid_x)
        valid_yy, valid_xx = tf.cast(grid[1], tf.bool), tf.cast(grid[0], tf.bool)
        valid = tf.math.logical_and(valid_xx, valid_yy)
        valid = tf.repeat(valid[:, None], num_base_anchors, axis=-1)
        valid = tf.reshape(valid, [-1])
        return valid


    def __repr__(self):
        indent_str = '  ' * 2
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str
