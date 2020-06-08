# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, RPNTestMixin 


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.
    
    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)


    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def call(self, inputs, training=True):
        if training:
            imgs, img_metas, gt_bboxes, gt_labels = inputs
        else:
            imgs, img_metas = inputs
        x = self.extract_feat(imgs)
        cls_scores_list, deltas_list = self.bbox_head(x, is_training=training)
        if training:

            class_loss, bbox_loss = self.bbox_head.loss(cls_scores_list, deltas_list, gt_bboxes, gt_labels, img_metas)
            losses_dict = {
                'class_loss': class_loss,
                'bbox_loss': bbox_loss
            }
            return losses_dict
        else:
            detections_dict = {}
            # cls_scores are probabilities (passed through sigmoid)
            detections_list = self.bbox_head.get_proposals(cls_scores_list, deltas_list, img_metas)
            detections_dict = {
                    'bboxes': detections_list[0][0],
                    'labels': detections_list[0][1],
                    'scores': detections_list[0][2]
            }
            return detections_dict

