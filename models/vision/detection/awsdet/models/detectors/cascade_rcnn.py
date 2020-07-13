# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from ..registry import DETECTORS
from .two_stage import TwoStageDetector
import tensorflow as tf

@DETECTORS.register_module
class CascadeRCNN(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 bbox_roi_extractor=None,
                 shared_head=None,
                 pretrained=None):
        super(CascadeRCNN, self).__init__(
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
    
    def init_weights(self):
        super(CascadeRCNN, self).init_weights(self.pretrained)
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
            gt_boxes, gt_class_ids = None, None

        C2, C3, C4, C5 = self.backbone(imgs, training=training)

        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5], training=training)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]

        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(rpn_feature_maps, training=training)

        proposals_list = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas, training=training)

        rcnn_class_logits, rcnn_probs, rcnn_deltas, rcnn_target_matches, rcnn_target_deltas, inside_weights, outside_weights, rois_list = \
            self.bbox_head(proposals_list, gt_boxes, gt_class_ids, img_metas, rcnn_feature_maps, training=training)

        if training:
            rpn_inputs = (rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas)
            rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(*rpn_inputs)

            losses_dict = {
                'rpn_class_loss': rpn_class_loss,
                'rpn_bbox_loss': rpn_bbox_loss
            }

            
            losses_dict = _get_rcnn_losses(rcnn_class_logits,
                                           rcnn_deltas, 
                                           rcnn_target_matches,
                                           rcnn_target_deltas, 
                                           inside_weights,
                                           outside_weights,
                                           losses_dict)
            
            return losses_dict
        else:
            detections_dict = {}
            # AS: currently we limit eval to 1 image bs per GPU - TODO: extend to multiple
            # detections_list will, at present, have len 1
            
            detections_list = self.bbox_head.bbox_heads[-1].get_bboxes(rcnn_probs[-1], rcnn_deltas[-1], rois_list, img_metas)
            detections_dict = {
                    'bboxes': detections_list[0][0],
                    'labels': detections_list[0][1],
                    'scores': detections_list[0][2]
            }
            
            return detections_dict
    
    def _get_rcnn_losses(self, 
                         rcnn_class_logits, 
                         rcnn_deltas, 
                         rcnn_target_matches, 
                         rcnn_target_deltas, 
                         inside_weights, 
                         outside_weights, 
                         losses_dict):
        for i in range(self.bbox_head.num_stages):
            rcnn_inputs = (rcnn_class_logits[i], rcnn_deltas[i], rcnn_target_matches[i],
                rcnn_target_deltas[i], inside_weights[i], outside_weights[i]) 
            rcnn_class_loss, rcnn_bbox_loss = self.bbox_head.bbox_heads[i].loss(rcnn_inputs)

            # we scale the losses of each stage differently.
            # TODO Figure out clean way to combine the stage_loss_weights variable with 
            # the batch_processor since batch_processer has a loss_weights param. This
            # is only a temporary solution.
            losses_dict['rcnn_class_loss_%d' % i] = rcnn_class_loss * self.bbox_head.stage_loss_weights[i]
            losses_dict['rcnn_bbox_loss_%d' % i] = rcnn_bbox_loss * self.bbox_head.stage_loss_weights[i]
        return losses_dict