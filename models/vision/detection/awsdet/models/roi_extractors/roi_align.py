# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
from awsdet.models.utils.misc import calc_pad_shapes
from ..registry import ROI_EXTRACTORS


# modification of aligned crop and resize from tensorpack
@tf.function(experimental_relax_shapes=True)
def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
    """
    The way tf.image.crop_and_resize works (with normalized box):
    Initial point (the value of output[0]): x0_box * (W_img - 1)
    Spacing: w_box * (W_img - 1) / (W_crop - 1)
    Use the above grid to bilinear sample.
    However, what we want is (with fpcoor box):
    Spacing: w_box / W_crop
    Initial point: x0_box + spacing/2 - 0.5
    (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
     (0.0, 0.0) is the same as pixel value (0, 0))
    This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
    Returns:
        y1x1y2x2
    """
    y0, x0, y1, x1 = tf.split(boxes, 4, axis=1)

    spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
    spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

    imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
    nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
    ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

    nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
    nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

    return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

@tf.function(experimental_relax_shapes=True)
def crop_and_resize(image, boxes, box_ind, image_shape, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
    Args:
        image: NHWC
        boxes: nx4, y1x1y2x2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        boxes = boxes + 1

    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    ret = tf.image.crop_and_resize(image, boxes, tf.cast(box_ind, tf.int32),
                                   crop_size=[crop_size, crop_size],
                                   method='bilinear')
    return ret


@ROI_EXTRACTORS.register_module
class PyramidROIAlign(tf.keras.Model):
    def __init__(self, pool_shape, pool_type='max', use_tf_crop_and_resize=True, **kwargs):
        '''
        Implements ROI Pooling on multiple levels of the feature pyramid.

        Attributes
        ---
            pool_shape: (height, width) of the output pooled regions.
                Example: (7, 7)
        '''
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pre_pool_shape = tuple([2*x for x in pool_shape])
        if pool_type == 'max':
            self._pool = layers.MaxPool2D(padding='same')
        elif pool_type == 'avg':
            self._pool = layers.AveragePooling2D(padding='same')
        self.use_tf_crop_and_resize = use_tf_crop_and_resize

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True):
        ''' 
        Args
        ---
            rois_list: list of [num_rois, (y1, x1, y2, x2)] in regular coordinates.
            feature_map_list: List of [batch, height, width, channels].
                feature maps from different levels of the pyramid.
            img_metas: [batch_size, 11]

        Returns
        ---
            pooled_rois_list: list of [num_rois, pooled_height, pooled_width, channels].
                The width and height are those specific in the pool_shape in the layer
                constructor.
        '''
        rois_list, feature_map_list, img_metas = inputs
        pad_shapes = calc_pad_shapes(img_metas)

        H = pad_shapes[:, 0][0]
        W = pad_shapes[:, 1][0]
        num_rois_list = [tf.shape(rois)[0] for rois in rois_list]
        roi_indices = tf.concat([tf.ones(tf.shape(rois_list[i])[0], dtype=tf.int32)*i for i in range(len(rois_list))], axis=0)

        rois = tf.concat(rois_list, axis=0)

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(rois, 4, axis=1)
        h = tf.maximum(tf.constant(0., dtype=y1.dtype), y2 - y1)
        w = tf.maximum(tf.constant(0., dtype=x1.dtype), x2 - x1)

        # Equation 1 in the Feature Pyramid Networks paper.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        areas = tf.sqrt(w * h + 1e-8)
        log_2 = tf.cast(tf.math.log(2.), dtype=areas.dtype)
        roi_levels = tf.floor(4. + tf.math.log(areas / tf.constant(224.0, dtype=areas.dtype)) / log_2)
        roi_levels = tf.maximum(roi_levels, tf.ones_like(roi_levels, dtype=roi_levels.dtype) * 2) # min level 2
        roi_levels = tf.minimum(roi_levels, tf.ones_like(roi_levels, dtype=roi_levels.dtype) * 5) # max level 5
        roi_levels = tf.stop_gradient(tf.reshape(roi_levels, [-1]))

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled_rois = []
        roi_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_levels, level))
            level_rois = tf.gather_nd(rois, ix)

            # ROI indices for crop_and_resize.
            level_roi_indices = tf.gather_nd(roi_indices, ix)

            # Keep track of which roi is mapped to which level
            roi_to_level.append(ix)

            # Stop gradient propagation to ROI proposals
            level_rois = tf.stop_gradient(level_rois)
            level_roi_indices = tf.stop_gradient(level_roi_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_rois, pool_height, pool_width, channels]
            if self.use_tf_crop_and_resize:
                # normalize rois
                norm_level_rois = tf.cast(level_rois, tf.float32) / tf.stack([H, W, H, W])
                crops = tf.image.crop_and_resize(feature_map_list[i], norm_level_rois,
                                             box_indices=level_roi_indices,
                                             crop_size=self.pre_pool_shape,
                                             method='bilinear')
            else:
                crops = crop_and_resize(feature_map_list[i],
                                    level_rois,
                                    level_roi_indices,
                                    [H, W],
                                    self.pre_pool_shape[0])
            crops = self._pool(crops)
            pooled_rois.append(crops)

        # Pack pooled features into one tensor
        pooled_rois = tf.concat(pooled_rois, axis=0)

        # Pack roi_to_level mapping into one array and add another
        # column representing the order of pooled rois
        roi_to_level = tf.concat(roi_to_level, axis=0)
        num_rois = tf.shape(roi_to_level)[0]
        roi_range = tf.expand_dims(tf.range(num_rois), 1)
        roi_to_level = tf.concat([tf.cast(roi_to_level, tf.int32), roi_range], axis=1)

        # Rearrange pooled features to match the order of the original rois
        # Sort roi_to_level by batch then roi index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = roi_to_level[:, 0] * 100000 + roi_to_level[:, 1]
        # arrange in reverse the order
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(roi_to_level)[0], sorted=True).indices[::-1]
        ix = tf.gather(roi_to_level[:, 1], ix) # [2000]
        pooled_rois = tf.gather(pooled_rois, ix) # [2000, 7, 7, 256]
        # 2000 of [7, 7, 256]
        pooled_rois_list = tf.split(pooled_rois, num_rois_list, axis=0)
        return pooled_rois_list
