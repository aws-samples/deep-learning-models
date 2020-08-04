# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf

class MaskTarget:

    def __init__(self, max_fg=128,
                       num_rois=512,
                       mask_size=(28, 28)):
        self.max_fg_mask = tf.concat([tf.ones(max_fg), tf.zeros(num_rois-max_fg)], axis=0)
        self.max_fg = max_fg
        self.mask_size = mask_size

    def get_assignments(self, fg_assignments, rcnn_target_matchs, img_metas, dtype=tf.int32):
        """
        For a set of foreground assignments and rcnn matches, return
        three tensors 
        
        fg_assignments : Reduced foreground assignments, restricted to only those that
        can actually contain a foreground
        
        rcnn_target_reduced : Reduced RCNN targets containing labels among the foreground
        group
        
        valid_fg : 0/1 tensor with 1 for valid foreground value
        
        All tensors of shape (N X max_fg)
        """
        batch_size = tf.shape(img_metas)[0]
        fg_mask = tf.tile(self.max_fg_mask, [batch_size])
        fg_assignments = tf.boolean_mask(fg_assignments, fg_mask)
        rcnn_target_reduced = tf.boolean_mask(rcnn_target_matchs, fg_mask) - 1
        valid_fg = tf.cast(rcnn_target_reduced>=0, tf.float32)
        rcnn_target_reduced = tf.keras.activations.relu(rcnn_target_reduced)
        fg_reduced = tf.cast(fg_assignments, dtype)
        rcnn_target_reduced = tf.cast(rcnn_target_reduced, dtype)
        return fg_reduced, rcnn_target_reduced, valid_fg

    def compute_offset(self, fg_assignments, gt_masks):
        """
        Compute the offset of fg_assignment based on an image's position with a batch
        """
        batch_size = tf.shape(gt_masks)[0]
        num_masks = tf.shape(gt_masks)[1]
        offset = tf.tile(tf.range(batch_size), [self.max_fg])
        offset = tf.reshape(tf.reshape(offset, [batch_size, self.max_fg]), [self.max_fg, batch_size])
        offset = tf.reshape(tf.transpose(offset), [-1])*num_masks
        return fg_assignments + offset

    def slice_masks(self, rcnn_masks, mask_indices):
        """
        Given an output from the mask head, subset it 
        into slices that correspond to the predicted class
        """
        masks = tf.transpose(rcnn_masks, [0, 3, 1, 2])
        indices = tf.transpose(tf.stack([tf.range(tf.size(mask_indices)), mask_indices]))
        return tf.expand_dims(tf.gather_nd(masks, indices), axis=-1)

    def get_weights(self, valid_flgs, img_metas):        
        batch_size = tf.shape(img_metas)[0]
        batch_size = tf.cast(batch_size, valid_flgs.dtype)
        pixel_size = tf.cast(self.mask_size[0]*self.mask_size[1], valid_flgs.dtype)
        weights = (valid_flgs/(tf.reduce_sum(valid_flgs)*pixel_size))*batch_size
        return weights

    def crop_masks(self, gt_masks, fg_rois_list, fg_offsets):
        """
        Given a set of ground truth masks, slice and subdivide to match
        ground truths to predictions
        """
        H = tf.shape(gt_masks)[2]
        W = tf.shape(gt_masks)[3]
        gt_masks = tf.reshape(gt_masks, [-1, H, W, 1])
        fg_rois = tf.concat(fg_rois_list, axis=0)
        norm_fg_rois = fg_rois / tf.cast(tf.stack([H, W, H, W]), fg_rois.dtype)
        crops = tf.image.crop_and_resize(image=gt_masks,
                                         boxes=norm_fg_rois,
                                         box_indices=fg_offsets,
                                         crop_size=self.mask_size,
                                         method='nearest')
        return crops

    def get_mask_targets(self, gt_masks, fg_assignments, rcnn_target_matchs, fg_rois_list, img_metas):
        fg_reduced, fg_targets, valid_flgs = self.get_assignments(fg_assignments, rcnn_target_matchs, img_metas)
        fg_offset = self.compute_offset(fg_reduced, gt_masks)
        mask_crops = self.crop_masks(gt_masks, fg_rois_list, fg_offset)
        weights = self.get_weights(valid_flgs, img_metas)
        return mask_crops, fg_targets, weights 
