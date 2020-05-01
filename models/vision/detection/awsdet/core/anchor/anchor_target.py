# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from awsdet.core.bbox import geometry, transforms
from awsdet.models.utils.misc import trim_zeros


class AnchorTarget:
    """
    for every generated anchors boxes: [326393, 4],
    create its rpn_target_matchs and rpn_target_matchs
    which is used to train RPN network.
    """
    def __init__(self,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        '''
        Compute regression and classification targets for anchors.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RPN.
            target_stds: [4]. Bounding box refinement standard deviation for RPN.
            num_rpn_deltas: int. Maximal number of Anchors per image to feed to rpn heads.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rpn_deltas = num_rpn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr

    def build_targets(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''
        Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image 
                coordinates. batch_size = 1 usually
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.

        Returns
        ---
            rpn_target_matchs: [batch_size, num_anchors] matches between anchors and GT boxes.
                1 = positive anchor, -1 = negative anchor, 0 = neutral anchor
            rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, log(dh), log(dw))] 
                Anchor bbox deltas.
        '''
        # start = tf.timestamp()
        rpn_target_matchs = []
        rpn_target_deltas = []
        rpn_inside_weights = []
        rpn_outside_weights = []
        num_imgs = gt_class_ids.shape[0] # batch size
        for i in range(num_imgs):
            target_match, target_delta, bbox_inside_weights, bbox_outside_weights = self._build_single_target(anchors, valid_flags[i], gt_boxes[i], gt_class_ids[i])
            rpn_target_matchs.append(target_match)
            rpn_target_deltas.append(target_delta)
            rpn_inside_weights.append(bbox_inside_weights)
            rpn_outside_weights.append(bbox_outside_weights)
        rpn_target_matchs = tf.concat(rpn_target_matchs, axis=0)
        rpn_target_deltas = tf.concat(rpn_target_deltas, axis=0)
        rpn_inside_weights = tf.concat(rpn_inside_weights, axis=0)
        rpn_outside_weights = tf.concat(rpn_outside_weights, axis=0)
        # tf.print('build targets took:', tf.timestamp() - start)
        return rpn_target_matchs, rpn_target_deltas, rpn_inside_weights, rpn_outside_weights

    @tf.function(experimental_relax_shapes=True) #TODO: revisit implementation to reduce retracing
    def _build_single_target(self, all_anchors, valid_flags, gt_bboxes, gt_class_ids):

        gt_bboxes, _ = trim_zeros(gt_bboxes)
        total_anchors = all_anchors.get_shape().as_list()[0]

        # 1. Filter anchors to valid area
        selected_anchor_idx = tf.where(tf.equal(valid_flags, 1))[:, 0]
        anchors = tf.gather(all_anchors, selected_anchor_idx)
        gt_bboxes = tf.cast(gt_bboxes, anchors.dtype)
        # 2. Find IoUs
        target_matchs = -tf.ones((tf.shape(anchors)[0],), tf.int32)
        overlaps = geometry.compute_overlaps(anchors, gt_bboxes)  # [anchors_size, gt_bboxes_size]
        argmax_overlaps = tf.argmax(overlaps, axis=1, output_type=tf.int32)
        max_overlaps = tf.reduce_max(overlaps, axis=1)
        gt_max_overlaps = tf.reduce_max(overlaps, axis=0)
        gt_argmax_overlaps = tf.where(tf.equal(overlaps, gt_max_overlaps))[:, 0]

        # Assign labels
        bg_cond = tf.math.less(max_overlaps, self.neg_iou_thr)
        target_matchs = tf.where(bg_cond, tf.zeros_like(target_matchs), target_matchs)
        gt_indices = tf.expand_dims(gt_argmax_overlaps, axis=1)
        gt_labels = tf.ones(tf.shape(gt_indices)[0], dtype=tf.int32)
        target_matchs = tf.tensor_scatter_nd_update(target_matchs, gt_indices, gt_labels)
        fg_cond = tf.math.greater_equal(max_overlaps, self.pos_iou_thr)
        target_matchs = tf.where(fg_cond, tf.ones_like(target_matchs), target_matchs)

        # Sample selected if more than that required
        fg_inds = tf.where(tf.equal(target_matchs, 1))[:, 0]
        max_pos_samples = tf.cast(self.positive_fraction * self.num_rpn_deltas, tf.int32)
        if tf.greater(tf.size(fg_inds), max_pos_samples):
            fg_inds = tf.random.shuffle(fg_inds)
            disable_inds = fg_inds[max_pos_samples:]
            fg_inds = fg_inds[:max_pos_samples]
            disable_inds = tf.expand_dims(disable_inds, axis=1)
            disable_labels = -tf.ones(tf.shape(disable_inds)[0], dtype=tf.int32)
            target_matchs = tf.tensor_scatter_nd_update(target_matchs, disable_inds, disable_labels)
        num_bg = self.num_rpn_deltas - tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32))
        bg_inds = tf.where(tf.equal(target_matchs, 0))[:, 0]
        if tf.greater(tf.size(bg_inds), num_bg):
            bg_inds = tf.random.shuffle(bg_inds)
            disable_inds = bg_inds[num_bg:]
            bg_inds = bg_inds[:num_bg]
            disable_inds = tf.expand_dims(disable_inds, axis=1)
            disable_labels = -tf.ones(tf.shape(disable_inds)[0], dtype=tf.int32)
            target_matchs = tf.tensor_scatter_nd_update(target_matchs, disable_inds, disable_labels)
        # tf.print('anchor target generated %d fgs and %d bgs.' % (tf.size(fg_inds), tf.size(bg_inds)))

        # Calculate deltas for chosen targets based on GT
        bboxes_targets = transforms.bbox2delta(anchors, tf.gather(gt_bboxes, argmax_overlaps),
                                                       target_means=self.target_means,
                                                       target_stds=self.target_stds)

        # Regression weights
        bbox_inside_weights = tf.zeros((tf.shape(anchors)[0], 4), dtype=tf.float32)
        match_indices = tf.where(tf.equal(target_matchs, 1))
        updates = tf.ones([tf.shape(match_indices)[0], 4], bbox_inside_weights.dtype)
        bbox_inside_weights = tf.tensor_scatter_nd_update(bbox_inside_weights,
                                                match_indices, updates)

        bbox_outside_weights = tf.zeros((tf.shape(anchors)[0], 4), dtype=tf.float32)
        num_examples = tf.reduce_sum(tf.cast(target_matchs >= 0, bbox_outside_weights.dtype))
        out_indices = tf.where(target_matchs >= 0)
        updates = tf.ones([tf.shape(out_indices)[0], 4], bbox_outside_weights.dtype) * 1.0 / num_examples
        bbox_outside_weights = tf.tensor_scatter_nd_update(bbox_outside_weights,
                                                out_indices, updates)
        # for everything that is not selected fill with `fill` value
        return (tf.stop_gradient(_unmap(target_matchs, total_anchors, selected_anchor_idx, -1)),
               tf.stop_gradient(_unmap(bboxes_targets, total_anchors, selected_anchor_idx, 0)),
               tf.stop_gradient(_unmap(bbox_inside_weights, total_anchors, selected_anchor_idx, 0)),
               tf.stop_gradient(_unmap(bbox_outside_weights, total_anchors, selected_anchor_idx, 0)))

def _unmap(data, count, inds, fill=0):
    """
    Fill data locations not in inds by fill value
    :param data:
    :param count:
    :param inds:
    :param fill:
    :return:
    """
    inds = tf.expand_dims(inds, axis=1)
    if len(data.shape) == 1:
        ret = tf.ones([count], dtype=data.dtype) * fill
        ret = tf.tensor_scatter_nd_update(ret, inds, data)
    else:
        ret = tf.repeat(tf.expand_dims(tf.ones(tf.shape(data)[1:], data.dtype), axis=0), count, axis=0) * fill
        ret = tf.tensor_scatter_nd_update(ret, inds, data)
    return ret

