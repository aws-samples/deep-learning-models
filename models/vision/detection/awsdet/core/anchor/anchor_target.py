# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from awsdet.core.bbox import geometry, transforms
from awsdet.models.utils.misc import trim_zeros
from awsdet.models.utils.misc import calc_img_shapes

class AnchorTarget:

    def __init__(self,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_samples=-1, # -1 disables sampling of targets
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        """
        Compute regression and classification targets for anchors.
        
        Args:
            target_means: Bounding box refinement mean
            target_stds: Bounding box refinement standard deviation
            num_samples (int): Number of anchors (sampled) per image. -1 disables sampling, e. g. RetinaNet
            positive_fraction (float): (Upper limit on) fraction of `num_samples` that need to be FG anchors
            pos_iou_thr (float): decides the IOU above which anchor is deemed to be FG
            neg_iou_thr(float): decides the IOU below which anchor is assigned to BG
        """
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr


    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_labels_list=None,
                    unmap_outputs=True):
        """
        Args:
            anchor_list List[List]]: anchors per level per image
            valid_flag_list List[List]]: valid flags corresponding to anchors in `anchor_list`
            gt_bboxes_list List: Ground truth boxes per image
            img_metas List:
            gt_labels_list List: GT labels corresponding to GT bboxes - these are assigned to target matches. Pass None for RPN.
            unmap_outputs:
        Returns:
            Tuple:
                target_matches_list (list): Target labels per image (corresponding to flattened anchors for FPN) 
                target_deltas_list (list): Target deltas per image corresponding to the target matches
                inside_weights_list (list): Weights (same shape as deltas) that are assigned to deltas that correspond to FG
                outside_weights_list (list): Weights (same shape as deltas) that are assigned to deltas that correspond to FG + BG
                    Used to provide a normalizing factor for bbox regression loss calculation. TODO: revisit this implementation
                total_pos_anchors (int): these are used later in loss calculation
                total_neg_anchors (int)
        """
        num_imgs = len(img_metas)
        num_level_anchors = [tf.shape(anchors)[0] for anchors in anchor_list[0]]
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(tf.concat(anchor_list[i], axis=0))
            concat_valid_flag_list.append(tf.concat(valid_flag_list[i], axis=0))
        target_matches_list = []
        target_deltas_list = []
        inside_weights_list = []
        outside_weights_list = []
        total_pos_anchors = 0
        total_neg_anchors = 0
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        img_shapes = calc_img_shapes(img_metas)
        for img_idx in range(num_imgs):
            anchors = concat_anchor_list[img_idx]
            valid_flags = concat_valid_flag_list[img_idx]
            gt_bboxes = gt_bboxes_list[img_idx]
            gt_labels = gt_labels_list[img_idx]
            img_shape = img_shapes[img_idx]
            (target_matches, target_deltas, bbox_inside_weights,
                    bbox_outside_weights, num_pos, num_neg) = self._get_targets_single(anchors,
                            valid_flags, gt_bboxes, gt_labels, img_shape)
            target_matches_list.append(target_matches)
            target_deltas_list.append(target_deltas)
            inside_weights_list.append(bbox_inside_weights)
            outside_weights_list.append(bbox_outside_weights)
            total_pos_anchors += num_pos
            total_neg_anchors += num_neg
        return (target_matches_list, target_deltas_list, inside_weights_list,
                outside_weights_list, total_pos_anchors, total_neg_anchors)


    def _anchor_inside_flags(self, flat_anchors, valid_flags, img_shape):
        img_h = img_shape[0]
        img_w = img_shape[1]
        inside_flags = tf.cast(valid_flags, tf.bool)
        # y x y x
        cond1 = tf.math.logical_and((flat_anchors[:, 0] >= 0.0), (flat_anchors[:, 1] >= 0.0))
        cond2 = tf.math.logical_and((flat_anchors[:, 2] < img_h),(flat_anchors[:, 3] < img_w))
        cond = tf.math.logical_and(cond1, cond2)
        return tf.math.logical_and(valid_flags, cond)


    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_labels,
                            img_shape,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in
            a single image.
        Args:
            flat_anchors: Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags: Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes: Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels: Ground truth labels of each box, shape (num_gts,). If not None then assign
                these labels to positive anchors
            img_shape: shape of the image (unpadded)
            unmap_outputs: Whether to map outputs back to the original
                set of anchors.
        Returns:
            target_matches: (num_anchors,) 1 = positive anchor, -1 = negative anchor, 0 = neutral anchor 
            bboxes_targets: (num_anchors, 4)
            bbox_inside_weights: (num_anchors, 4)
            bbox_outside_weights: (num_anchors, 4)
        """
        gt_bboxes, _ = trim_zeros(gt_bboxes)

        # 1. Filter anchors to valid area
        inside_flags = self._anchor_inside_flags(flat_anchors, valid_flags, img_shape)
        # TODO: handle scenario where all flags are False
        anchors = tf.boolean_mask(flat_anchors, inside_flags)
        num_anchors = tf.shape(flat_anchors)[0]

        # 2. Find IoUs
        num_valid_anchors = tf.shape(anchors)[0]
        target_matches = -tf.ones((num_valid_anchors,), tf.int32)
        overlaps = geometry.compute_overlaps(anchors, gt_bboxes)
        # a. best GT index for each anchor
        argmax_overlaps = tf.argmax(overlaps, axis=1, output_type=tf.int32)
        max_overlaps = tf.reduce_max(overlaps, axis=1)
        # b. best anchor index for each GT (non deterministic in case of ties)
        gt_argmax_overlaps = tf.argmax(overlaps, axis=0, output_type=tf.int32) # tf.where(tf.equal(overlaps, gt_max_overlaps))[:, 0]

        # 3. Assign labels
        bg_cond = tf.math.less(max_overlaps, self.neg_iou_thr)
        fg_cond = tf.math.greater_equal(max_overlaps, self.pos_iou_thr)
        target_matches = tf.where(bg_cond, tf.zeros_like(target_matches), target_matches)
        gt_indices = tf.expand_dims(gt_argmax_overlaps, axis=1)
        if gt_labels is None: # RPN will have gt labels set to None
            gt_labels = tf.ones(tf.shape(gt_indices)[0], dtype=tf.int32)
            #TODO check impact of next 2 lines
            target_matches = tf.tensor_scatter_nd_update(target_matches, gt_indices, gt_labels) # note that in the case of one label matching multiple anchors the last one wins (is this okay???)
            target_matches = tf.where(fg_cond, tf.ones_like(target_matches), target_matches)
        else:
            gt_labels = gt_labels[:tf.shape(gt_indices)[0]] # get rid of padded labels (-1)
            target_matches = tf.where(fg_cond, tf.gather(gt_labels, argmax_overlaps), target_matches)

        # 4. Sample selected if we have greater number of candidates than needed by 
        #    config (only if num_samples > 0, e.g. in two stage)
        if self.num_samples > 0:
            fg_inds = tf.where(tf.equal(target_matches, 1))[:, 0]
            max_pos_samples = tf.cast(self.positive_fraction * self.num_samples, tf.int32)
            if tf.greater(tf.size(fg_inds), max_pos_samples):
                fg_inds = tf.random.shuffle(fg_inds)
                disable_inds = fg_inds[max_pos_samples:]
                fg_inds = fg_inds[:max_pos_samples]
                disable_inds = tf.expand_dims(disable_inds, axis=1)
                disable_labels = -tf.ones(tf.shape(disable_inds)[0], dtype=tf.int32)
                target_matches = tf.tensor_scatter_nd_update(target_matches, disable_inds, disable_labels)
            num_fg = tf.reduce_sum(tf.cast(tf.equal(target_matches, 1), tf.int32))
            num_bg = self.num_samples - num_fg 
            bg_inds = tf.where(tf.equal(target_matches, 0))[:, 0]
            if tf.greater(tf.size(bg_inds), num_bg):
                bg_inds = tf.random.shuffle(bg_inds)
                disable_inds = bg_inds[num_bg:]
                bg_inds = bg_inds[:num_bg]
                disable_inds = tf.expand_dims(disable_inds, axis=1)
                disable_labels = -tf.ones(tf.shape(disable_inds)[0], dtype=tf.int32)
                target_matches = tf.tensor_scatter_nd_update(target_matches, disable_inds, disable_labels)

        # 5. Calculate deltas for chosen targets based on GT (encode)
        bboxes_targets = transforms.bbox2delta(anchors, tf.gather(gt_bboxes, argmax_overlaps),
                                                       target_means=self.target_means,
                                                       target_stds=self.target_stds)

        # Regression weights
        bbox_inside_weights = tf.zeros((tf.shape(anchors)[0], 4), dtype=tf.float32)
        # match_indices = tf.where(tf.equal(target_matches, 1))
        match_indices = tf.where(tf.math.greater(target_matches, 0))

        updates = tf.ones([tf.shape(match_indices)[0], 4], bbox_inside_weights.dtype)
        bbox_inside_weights = tf.tensor_scatter_nd_update(bbox_inside_weights,
                                                match_indices, updates)

        bbox_outside_weights = tf.zeros((tf.shape(anchors)[0], 4), dtype=tf.float32)
        if self.num_samples > 0:
            num_examples = tf.reduce_sum(tf.cast(target_matches >= 0, bbox_outside_weights.dtype))
        else:
            num_examples = tf.reduce_sum(tf.cast(target_matches > 0, bbox_outside_weights.dtype))
            num_fg = num_examples
            num_bg = 0 # in RetinaNet we only care about positive anchors
        out_indices = tf.where(target_matches >= 0)
        updates = tf.ones([tf.shape(out_indices)[0], 4], bbox_outside_weights.dtype) * 1.0 / num_examples
        bbox_outside_weights = tf.tensor_scatter_nd_update(bbox_outside_weights,
                                                out_indices, updates)
        # for everything that is not selected fill with `fill` value
        selected_anchor_idx = tf.where(inside_flags)[:, 0]
        return (tf.stop_gradient(_unmap(target_matches, num_anchors, selected_anchor_idx, -1)),
               tf.stop_gradient(_unmap(bboxes_targets, num_anchors, selected_anchor_idx, 0)),
               tf.stop_gradient(_unmap(bbox_inside_weights, num_anchors, selected_anchor_idx, 0)),
               tf.stop_gradient(_unmap(bbox_outside_weights, num_anchors, selected_anchor_idx, 0)),
               num_fg, num_bg)


def _unmap(data, count, inds, fill=0):
    """
    Fill data locations not in inds by fill value
    Args:
        data: used to update
        count: total length of tensor that will be updated with `data`
        inds: indices that indicate position in final tensor that will be updated (has same outer dim as data)
        fill: value that is put in all locations in the final tensor that is not pointed to by `inds`
    Returns:
        Updated tensor of outer dim `count` where locations not in `inds` are filled with `fill` value
    """
    inds = tf.expand_dims(inds, axis=1)
    if len(data.shape) == 1:
        ret = tf.ones([count], dtype=data.dtype) * fill
        ret = tf.tensor_scatter_nd_update(ret, inds, data)
    else:
        ret = tf.repeat(tf.expand_dims(tf.ones(tf.shape(data)[1:], data.dtype), axis=0), count, axis=0) * fill
        ret = tf.tensor_scatter_nd_update(ret, inds, data)
    return ret

