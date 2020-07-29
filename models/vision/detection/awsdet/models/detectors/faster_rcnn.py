# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from ..registry import DETECTORS

from .two_stage import TwoStageDetector
import os
import tensorflow as tf

from awsdet.core.bbox import bbox_target
#from awsdet.datasets import dali


@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
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
            neg_iou_thr=0.0)
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
            imgs, img_metas, gt_boxes, gt_class_ids = inputs
        else: # inference
            imgs, img_metas = inputs
        s0 = tf.timestamp()
        C2, C3, C4, C5 = self.backbone(imgs, training=training)
        s1 = tf.timestamp() 
        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5], training=training)
        s2 = tf.timestamp()
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(rpn_feature_maps, training=training)
        s3 = tf.timestamp()
        proposals_list = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas, training=training)
        s4 = tf.timestamp()
        if training: # get target value for these proposal target label and target delta
            rois_list, rcnn_target_matchs, rcnn_target_deltas, inside_weights, outside_weights = self.bbox_target.build_targets(proposals_list, gt_boxes, gt_class_ids, img_metas)
        else:
            rois_list = proposals_list
        s5 = tf.timestamp()
        # rois_list only contains coordinates, rcnn_feature_maps save the 5 features data=>[192,7,7,256]
        pooled_regions_list = self.bbox_roi_extractor(
            (rois_list, rcnn_feature_maps, img_metas), training=training)
        s6 = tf.timestamp()
        rcnn_class_logits, rcnn_probs, rcnn_deltas = self.bbox_head(pooled_regions_list, training=training)
        s7 = tf.timestamp()
        # tf.print('backbone', s1-s0)
        # tf.print('fpn', s2-s1)
        # tf.print('rpn head', s3-s2)
        # tf.print('proposal generation', s4-s3)
        # tf.print('roi target', s5-s4)
        # tf.print('pooling', s6-s5)
        # tf.print('roi head', s7-s6)
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
            return detections_dict
