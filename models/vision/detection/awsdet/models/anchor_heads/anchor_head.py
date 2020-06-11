# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
import functools
from awsdet.core.bbox import transforms
from awsdet.models.utils.misc import calc_pad_shapes
from awsdet.core.anchor.anchor_generator import AnchorGenerator
from awsdet.models.losses import losses
from ..registry import HEADS

@HEADS.register_module
class AnchorHead(tf.keras.Model):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).
    Args:
        num_classes (int): Number of categories including the background
            category
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """
    def __init__(self,
                 num_classes,
                 feat_channels=256,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 anchor_scales=None,
                 anchor_ratios=None,
                 anchor_strides=None,
                 target_means=None,
                 target_stds=None):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.target_means = target_means
        self.target_stds = target_stds
        self.anchor_generator = AnchorGenerator(
            scales=anchor_scales,
            ratios=anchor_ratios,
            strides=anchor_strides,
            octave_base_scale=octave_base_scale,
            scales_per_octave=scales_per_octave)
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()


    def _init_layers(self):
        '''
        This should be overridden by the subclass heads
        '''
        raise NotImplementedError


    def get_anchors(self, featmap_sizes, img_metas):
        """
        Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list): multi level feature map sizes
            img_metas (list): images metas in a batch
        Returns:
            anchor_list (list): anchors for each image in batch
            valid_flag_list (list): valid flags for each image
        """
        num_imgs = len(img_metas)
        # compute anchors only once per batch since backbone outputs on padded are of same dimensions
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        img_shapes = calc_pad_shapes(img_metas)
        for img_idx in range(num_imgs):
            multi_level_flags = self.anchor_generator.valid_flags(featmap_sizes, img_shapes[img_idx])
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list


    def loss(self,
                cls_scores,
                deltas,
                gt_boxes,
                gt_labels,
                img_metas):
        """
        Must be implemented by subclass now because of anchor target dependency
        TODO: build common when assigner and sampler functionality is done
        Args:

        Returns:
            
        """
        raise NotImplementedError


    def get_proposals(self, 
                        probs,
                        deltas,
                        img_metas,
                        with_probs=False,
                        training=True):
        """
        Get proposal bboxes for next stage or get boxes for prediction is single stage.
        
        Args:
            probs (list): List of probability scores for each scale level [N, H, W, num_anchors * num_classes]
            deltas (list): List of deltas for each scale [N, H, W, num_anchors * 4]
            img_metas: [N, 11]

        Returns:
            proposals_list: list of [num_proposals, (y1, x1, y2, x2)]
        """
        assert len(probs) == len(deltas)
        num_levels = len(probs)
        featmap_sizes = []
        for i in range(num_levels):
            feat_shape = tf.shape(probs[i])[1:3] # H, W
            featmap_sizes.append(feat_shape)

        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes)

        proposals_list = []
        img_shapes = calc_pad_shapes(img_metas)
        # for each image in batch generate proposals
        num_imgs = len(img_metas)
        for img_idx in range(num_imgs):
            img_probs = [tf.stop_gradient(probs[i][img_idx]) for i in range(num_levels)]
            img_deltas = [tf.stop_gradient(deltas[i][img_idx]) for i in range(num_levels)]
            img_shape = img_shapes[img_idx]
            proposals = self._get_proposals_single(img_probs, img_deltas,
                            mlvl_anchors, img_shape, with_probs, training)
            proposals_list.append(proposals)
        return proposals_list

    def _get_proposals_single(self,
                           probs_list,
                           deltas_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           with_probs,
                           training=True):
        """
        Transform outputs for a single batch item into labeled boxes.
        To be implemented by subclass
        """
        raise NotImplementedError

