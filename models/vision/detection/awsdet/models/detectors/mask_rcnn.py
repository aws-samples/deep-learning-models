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
from awsdet.models.mask_heads import mask_head
from awsdet.models.roi_extractors import roi_align
from awsdet.models.detectors.test_mixins import RPNTestMixin, BBoxTestMixin

from awsdet.core.bbox import bbox_target

@DETECTORS.register_module
class MaskRCNN(TwoStageDetector):
    
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(MaskRCNN, self).__init__(
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
        #TODO: delegate to assigner and sampler in the future
        self.bbox_target = bbox_target.ProposalTarget(
            target_means=self.bbox_head.target_means,
            target_stds=self.bbox_head.target_stds, 
            num_rcnn_deltas=512,
            positive_fraction=0.25,
            pos_iou_thr=0.5,
            neg_iou_thr=0.1,
            fg_assignments=True)
        self.count = 0
        
    def init_weights(self):
        super(MaskRCNN, self).init_weights(self.pretrained)
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
            imgs, img_metas, gt_boxes, gt_class_ids, gt_masks = inputs
        else: # inference
            imgs, img_metas = inputs
        C2, C3, C4, C5 = self.backbone(imgs, training=training)
        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5], training=training)
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(rpn_feature_maps, training=training)
        proposals_list = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas, training=training)
        if training:
            rois_list, rcnn_target_matchs, rcnn_target_deltas, \
            inside_weights, outside_weights, fg_assignments = \
            self.bbox_target.build_targets(proposals_list, gt_boxes, gt_class_ids, img_metas)
        else:
            rois_list = proposals_list
        pooled_regions_list = self.bbox_roi_extractor(
            (rois_list, rcnn_feature_maps, img_metas), training=training)
        rcnn_class_logits, rcnn_probs, rcnn_deltas = self.bbox_head(pooled_regions_list, training=training)
        if training:
            mask_pooled_regions_list = self.mask_roi_extractor(
                                    (rois_list, rcnn_feature_maps, img_metas), training=training)
            rcnn_masks = self.mask_head(mask_pooled_regions_list, training=training)
            rcnn_mask_loss = self.mask_head.mask_loss(rcnn_masks, rcnn_target_matchs, rois_list, 
                                                       fg_assignments, gt_masks, img_metas)
            #rcnn_mask_loss = self.mask_head.loss(rcnn_masks, fg_assignments, rcnn_target_matchs, 
            #                                     rois_list, gt_masks, img_metas)
            rpn_inputs = (rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas)
            rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(*rpn_inputs)
            rcnn_inputs = (rcnn_class_logits, rcnn_deltas, rcnn_target_matchs,
                rcnn_target_deltas, inside_weights, outside_weights) 
            rcnn_class_loss, rcnn_bbox_loss = self.bbox_head.loss(rcnn_inputs)
            losses_dict = {
                'rpn_class_loss': rpn_class_loss,
                'rpn_bbox_loss': rpn_bbox_loss,
                'rcnn_class_loss': rcnn_class_loss,
                'rcnn_bbox_loss': rcnn_bbox_loss,
                'mask_loss': rcnn_mask_loss
            }
            return losses_dict
        else:
            detections_dict = {}
            detections_list = self.bbox_head.get_bboxes(rcnn_probs, rcnn_deltas, rois_list, img_metas)
            detections_dict = {
                    'bboxes': detections_list[0][0],
                    'labels': detections_list[0][1],
                    'scores': detections_list[0][2]
            }
            mask_boxes = [tf.round(detections_dict['bboxes'])]
            mask_pooled_regions_list = self.mask_roi_extractor(
                                        (mask_boxes, 
                                         rcnn_feature_maps, img_metas), training=training)
            rcnn_masks = self.mask_head(mask_pooled_regions_list)[0]
            rcnn_masks = self.mask_head.gather_mask_predictions(rcnn_masks, detections_dict['labels'])
            detections_dict['masks'] = self.mask_head.mold_masks(rcnn_masks, mask_boxes[0], img_metas[0])
            return detections_dict
            