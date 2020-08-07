# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from ..registry import DETECTORS

from .two_stage import TwoStageDetector
import os
import tensorflow as tf
from awsdet.models.necks import fpn
from awsdet.models.anchor_heads import rpn_head
from awsdet.models.bbox_heads import bbox_head
from awsdet.models.roi_extractors import roi_align
from awsdet.models.detectors.test_mixins import RPNTestMixin, BBoxTestMixin

from awsdet.core.bbox import bbox_target
#from awsdet.datasets import dali
import numpy as np


@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 mask_roi_extractor=None,
                 mask_head=None,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.pretrained = pretrained
        self.mask = mask_head!=None
        #TODO: delegate to assigner and sampler in the future
        self.bbox_target = bbox_target.ProposalTarget(
            target_means=self.bbox_head.target_means,
            target_stds=self.bbox_head.target_stds, 
            num_rcnn_deltas=512,
            positive_fraction=0.25,
            pos_iou_thr=0.5,
            neg_iou_thr=0.1,
            fg_assignments=self.mask)
        self.count = 0

    def init_weights(self):
        super(FasterRCNN, self).init_weights(self.pretrained)
        if not self.pretrained:
            if hasattr(self.backbone, 'pretrained'):
                if not self.backbone.pretrained:
                    return
                else:
                    # check if backbone has weights
                    self.backbone.init_weights()
        else:
            #_, extension = os.path.splitext(self.pretrained)
            self.load_weights(self.pretrained) # , by_name=True)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True, use_dali=False):
        """
        :param inputs: [1, 1216, 1216, 3], [1, 11], [1, 14, 4], [1, 14]
        :param training:
        :return:
        """
        if use_dali:
            inputs = dali.dali_adapter(*inputs, training=training)
        if training: # training
            if self.mask:
                imgs, img_metas, gt_boxes, gt_class_ids, gt_masks = inputs
            else:
                imgs, img_metas, gt_boxes, gt_class_ids = inputs
        else: # inference
            imgs, img_metas = inputs
        s0 = tf.timestamp()
        # [1, 304, 304, 256] => [1, 152, 152, 512]=>[1,76,76,1024]=>[1,38,38,2048]
        C2, C3, C4, C5 = self.backbone(imgs, training=training)
        s1 = tf.timestamp() 
        # [1, 304, 304, 256] <= [1, 152, 152, 256]<=[1,76,76,256]<=[1,38,38,256]=>[1,19,19,256]
        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5], training=training)
        s2 = tf.timestamp()
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]
        # [1, 369303, 2] [1, 369303, 2], [1, 369303, 4], includes all anchors on pyramid level of features
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(rpn_feature_maps, training=training)
        s3 = tf.timestamp()
        # [369303, 4] => [215169, 4], valid => [6000, 4], performance =>[2000, 4],  NMS
        proposals_list = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas, training=training)
        s4 = tf.timestamp()
        if training: # get target value for these proposal target label and target delta
            if self.mask:
                rois_list, rcnn_target_matchs, rcnn_target_deltas, inside_weights, outside_weights, fg_assignments = \
                                        self.bbox_target.build_targets(proposals_list, gt_boxes, gt_class_ids, img_metas)
            else:
                rois_list, rcnn_target_matchs, rcnn_target_deltas, inside_weights, outside_weights = \
                                        self.bbox_target.build_targets(proposals_list, gt_boxes, gt_class_ids, img_metas)
        else:
            rois_list = proposals_list
        s5 = tf.timestamp()
        # rois_list only contains coordinates, rcnn_feature_maps save the 5 features data=>[192,7,7,256]
        pooled_regions_list = self.bbox_roi_extractor(
            (rois_list, rcnn_feature_maps, img_metas), training=training)
        s6 = tf.timestamp()
        # [192, 81], [192, 81], [192, 81, 4]
        rcnn_class_logits, rcnn_probs, rcnn_deltas = self.bbox_head(pooled_regions_list, training=training)
        s7 = tf.timestamp()
        # tf.print('backbone', s1-s0)
        # tf.print('fpn', s2-s1)
        # tf.print('rpn head', s3-s2)
        # tf.print('proposal generation', s4-s3)
        # tf.print('roi target', s5-s4)
        # tf.print('pooling', s6-s5)
        # tf.print('roi head', s7-s6)
        # if training use rpn outputs to compute mask regions
        if training and self.mask:
            fg_rois_list = self.mask_head.get_fg_rois_list(rois_list)
            mask_regions_list = self.mask_roi_extractor((fg_rois_list, rcnn_feature_maps, img_metas), training=training)
            gt_mask_crops, fg_targets, weights = \
                        self.mask_head.mask_target.get_mask_targets(gt_masks, fg_assignments, 
                                                                    rcnn_target_matchs, fg_rois_list, img_metas)
            rcnn_masks = self.mask_head(mask_regions_list)
            rcnn_masks = self.mask_head.mask_target.slice_masks(rcnn_masks, fg_targets)
            mask_loss = self.mask_head.loss(gt_mask_crops, rcnn_masks, weights)
        if training:
            s8 = tf.timestamp()
            rpn_inputs = (rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas)
            rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(*rpn_inputs)
            s9 = tf.timestamp()
            rcnn_inputs = (rcnn_class_logits, rcnn_deltas, rcnn_target_matchs,
                rcnn_target_deltas, inside_weights, outside_weights) 
            s10 = tf.timestamp()
            rcnn_class_loss, rcnn_bbox_loss = self.bbox_head.loss(rcnn_inputs)
            # tf.print('rpn loss', s9-s8)
            # tf.print('roi loss', s10-s9)
            losses_dict = {
                'rpn_class_loss': rpn_class_loss,
                'rpn_bbox_loss': rpn_bbox_loss,
                'rcnn_class_loss': rcnn_class_loss,
                'rcnn_bbox_loss': rcnn_bbox_loss
            }
            if self.mask:
                losses_dict['rcnn_mask_loss'] = mask_loss
            return losses_dict
        else:
            detections_dict = {}
            # AS: currently we limit eval to 1 image bs per GPU - TODO: extend to multiple
            # detections_list will, at present, have len 1
            detections_list = self.bbox_head.get_bboxes(rcnn_probs, rcnn_deltas, rois_list, img_metas)
            detections_dict = {
                    'bboxes': detections_list[0][0],
                    'labels': detections_list[0][1],
                    'scores': detections_list[0][2]
            }
            if self.mask:
                mask_boxes = [tf.round(detections_dict['bboxes'])]
                mask_pooled_regions_list = self.mask_roi_extractor(
                                        (mask_boxes, 
                                         rcnn_feature_maps, img_metas), training=training)
                rcnn_masks = self.mask_head(mask_pooled_regions_list)
                rcnn_masks = self.mask_head.mask_target.slice_masks(rcnn_masks, detections_dict['labels'] - 1)
                detections_dict['masks'] = self.mask_head.mold_masks(rcnn_masks, mask_boxes[0], img_metas[0])
            return detections_dict
