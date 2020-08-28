# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from awsdet.core.bbox import geometry, transforms
from awsdet.models.utils.misc import calc_pad_shapes, trim_zeros

class ProposalTarget:

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rcnn_deltas=512,
                 positive_fraction=0.25,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5,
                 num_classes=81,
                 reg_class_agnostic=False,
                 fg_assignments=False):
        """
        Compute regression and classification targets for proposals.
        
        Attributes:
            target_means: [4]. Bounding box refinement mean for RCNN.
            target_stds: [4]. Bounding box refinement standard deviation for RCNN.
            num_rcnn_deltas: int. Maximal number of RoIs per image to feed to bbox heads.

        """
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rcnn_deltas = num_rcnn_deltas
        self.positive_fraction = positive_fraction
        self._max_pos_samples = int(positive_fraction * num_rcnn_deltas)
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.fg_assignments = fg_assignments
            
    def build_targets(self, proposals_list, gt_boxes, gt_class_ids, img_metas):
        """
        Generate detection targets for images.
        Subsample proposals and generate target class IDs, bounding box deltas for each.
        
        Args:
            proposals_list: list of [num_proposals, (y1, x1, y2, x2)] in regular coordinates.
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
            img_metas: [batch_size, 11]
            
        Returns:
            rois_list: list of [num_rois, (y1, x1, y2, x2)] in normalized coordinates
            rcnn_target_matchs_list: list of [num_rois]. Integer class IDs.
            rcnn_target_deltas_list: list of [num_positive_rois, (dy, dx, log(dh), log(dw))].
            
        Note that self.num_rcnn_deltas >= num_rois > num_positive_rois. And different 
           images in one batch may have different num_rois and num_positive_rois.
        """
 
        pad_shapes = calc_pad_shapes(img_metas) 
        batch_size = img_metas.shape[0]
        
        rois_list = []
        rcnn_target_matchs_list = []
        rcnn_target_deltas_list = []
        inside_weights_list = []
        outside_weights_list = []
        fg_assignments_list = []
        for i in range(img_metas.shape[0]):
            rois, target_matchs, target_deltas, inside_weights, outside_weights, fg_assignments = self._build_single_target(
                                                                                    proposals_list[i], 
                                                                                    gt_boxes[i],
                                                                                    gt_class_ids[i],
                                                                                    pad_shapes[i])
            rois_list.append(rois)
            rcnn_target_matchs_list.append(target_matchs)
            rcnn_target_deltas_list.append(target_deltas)
            inside_weights_list.append(inside_weights)
            outside_weights_list.append(outside_weights)
            fg_assignments_list.append(fg_assignments)

        # rois = tf.concat(rois_list, axis=0)
        rcnn_target_matchs = tf.concat(rcnn_target_matchs_list, axis=0)
        rcnn_target_deltas = tf.concat(rcnn_target_deltas_list, axis=0)
        inside_weights = tf.concat(inside_weights_list, axis=0)
        outside_weights = tf.concat(outside_weights_list, axis=0)
        fg_assignments = tf.concat(fg_assignments_list, axis=0)
        # TODO: concat proposals list and rois_list 
        if self.fg_assignments:
            return (rois_list, rcnn_target_matchs, rcnn_target_deltas, 
                    inside_weights, outside_weights, fg_assignments)
        return (rois_list, rcnn_target_matchs, rcnn_target_deltas, 
                inside_weights, outside_weights)


    @tf.function(experimental_relax_shapes=True)
    def _build_single_target(self, proposals, gt_boxes, gt_class_ids, img_shape):
        '''
        Args
        ---
            proposals: [num_proposals, (y1, x1, y2, x2)] in regular coordinates.
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
            gt_class_ids: [num_gt_boxes]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            
        Returns
        ---
            rois: [num_rois, (y1, x1, y2, x2)]
            target_matchs: [num_positive_rois]
            target_deltas: [num_positive_rois, (dy, dx, log(dh), log(dw))]
        '''
        # remove padded proposals and gt boxes if any
        proposals, _ = trim_zeros(proposals)
        gt_boxes, non_zeros = trim_zeros(gt_boxes)
        gt_boxes = tf.cast(gt_boxes, proposals.dtype)
        gt_labels = tf.boolean_mask(gt_class_ids, non_zeros)
        proposals_gt = tf.concat([proposals, gt_boxes], axis=0)

        iou = geometry.compute_overlaps(proposals_gt, gt_boxes)
        max_overlaps = tf.reduce_max(iou, axis=1)
        gt_assignment = tf.argmax(iou, axis=1)
        labels = tf.gather(gt_labels, gt_assignment)

        # get FG and BG
        fg_inds = tf.where(max_overlaps >= self.pos_iou_thr)[:, 0]
        bg_inds = tf.where(tf.logical_and(max_overlaps < self.pos_iou_thr,
                                          max_overlaps >= self.neg_iou_thr))[:, 0]
        # filter FG/BG
        if tf.size(fg_inds) > self._max_pos_samples:
            fg_inds = tf.random.shuffle(fg_inds)[:self._max_pos_samples]
        remaining = self.num_rcnn_deltas - tf.size(fg_inds)
        num_bg = tf.size(bg_inds)
        if tf.greater_equal(num_bg, remaining):
            bg_inds = tf.random.shuffle(bg_inds)[:remaining]
        else:
            # sample with replacement from very poor overlaps if number of backgrounds is not enough
            bg_inds = tf.where(max_overlaps < self.pos_iou_thr)[:, 0]
            bg_inds = tf.random.shuffle(bg_inds)[:remaining]
            num_bg = tf.size(bg_inds)
            while remaining > num_bg:
                dups = remaining - num_bg
                dup_bgs = tf.random.shuffle(bg_inds)[:dups]
                bg_inds = tf.concat([bg_inds, dup_bgs], axis=0)
                num_bg = tf.size(bg_inds)

        # tf.print('proposal target generated %d fgs and %d bgs.' % (tf.size(fg_inds), tf.size(bg_inds)))

        keep_inds = tf.concat([fg_inds, bg_inds], axis=0)
        final_rois = tf.gather(proposals_gt, keep_inds)  # rois[keep_inds]
        final_labels = tf.gather(labels, keep_inds)  # labels[keep_inds]
        zero_indices = tf.expand_dims(tf.range(tf.size(fg_inds), tf.size(keep_inds), dtype=tf.int32), axis=1)
        zero_labels = tf.zeros(tf.shape(zero_indices)[0], dtype=tf.int32)
        final_labels = tf.tensor_scatter_nd_update(final_labels, zero_indices, zero_labels)

        # inside weights - positive examples are set, rest are zeros
        bbox_inside_weights = tf.zeros((tf.size(keep_inds), self.num_classes, 4), dtype=tf.float32)
        if tf.size(fg_inds) > 0:
            if self.reg_class_agnostic:
                cur_index = tf.transpose(tf.stack([tf.range(tf.size(fg_inds)), tf.zeros(tf.size(fg_inds), dtype=tf.int32)]))
            else:
                cur_index = tf.stack([tf.range(tf.size(fg_inds)), tf.gather(labels, fg_inds)], axis=1)
            bbox_inside_weights = tf.tensor_scatter_nd_update(bbox_inside_weights,
                                                       cur_index,
                                                       tf.ones([tf.size(fg_inds), 4], bbox_inside_weights.dtype))
        bbox_inside_weights = tf.reshape(bbox_inside_weights, [-1, self.num_classes * 4])

        final_bbox_targets = tf.zeros((tf.size(keep_inds), self.num_classes, 4), dtype=tf.float32)
        if tf.size(fg_inds) > 0:
            bbox_targets = transforms.bbox2delta(
                tf.gather(final_rois, tf.range(tf.size(fg_inds))),
                tf.gather(gt_boxes, tf.gather(gt_assignment, fg_inds)),
                target_stds=self.target_stds, target_means=self.target_means)
            if self.reg_class_agnostic:
                final_bbox_targets = tf.tensor_scatter_nd_update(
                                        final_bbox_targets,
                                        tf.transpose(tf.stack([tf.range(tf.size(fg_inds)),
                                        tf.zeros(tf.size(fg_inds), dtype=tf.int32)])),
                                        bbox_targets)
            else:
                final_bbox_targets = tf.tensor_scatter_nd_update(
                                        final_bbox_targets,
                                        tf.stack([tf.range(tf.size(fg_inds)),
                                        tf.gather(labels, fg_inds)], axis=1), bbox_targets)
        final_bbox_targets = tf.reshape(final_bbox_targets, [-1, self.num_classes * 4])

        bbox_outside_weights = tf.ones_like(bbox_inside_weights, dtype=bbox_inside_weights.dtype) * 1.0 / self.num_rcnn_deltas
        fg_assignments = tf.gather(gt_assignment, keep_inds)
        return (tf.stop_gradient(final_rois), tf.stop_gradient(final_labels), tf.stop_gradient(final_bbox_targets),
               tf.stop_gradient(bbox_inside_weights), tf.stop_gradient(bbox_outside_weights), tf.stop_gradient(fg_assignments))


