# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
import functools
import numpy as np
from .anchor_head import AnchorHead
from awsdet.core.bbox import transforms
from awsdet.models.utils.misc import calc_pad_shapes
from awsdet.core.anchor import anchor_target
from awsdet.models.losses import losses
from ..registry import HEADS

@HEADS.register_module
class RetinaHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 stacked_convs=4,
                 feat_channels=256,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 target_means=[.0, .0, .0, .0],
                 target_stds=[1.0, 1.0, 1.0, 1.0],
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.4,
                 alpha=0.25,
                 gamma=2.0,
                 label_smoothing=0.05,
                 num_pre_nms=1000,
                 min_confidence=0.005,
                 nms_threshold=0.5,
                 max_instances=100,
                 soft_nms_sigma=0.5,
                 weight_decay=1e-5
                 ):
        '''
        Anchor head of as described in `RetinaNet <https://arxiv.org/pdf/1708.02002.pdf>`_.
        The head contains two subnetworks. The first classifies anchor boxes and
        the second regresses deltas for the anchors.

        Attributes
        ---
            anchor_scales: 1D array of anchor sizes in pixels.
            anchor_ratios: 1D array of anchor ratios of width/height.
            anchor_strides: Stride of the feature map relative 
                to the image in pixels.
            max_instances: int. bboxes kept after non-maximum 
                suppression.
            nms_threshold: float. Non-maximum suppression threshold to 
                filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_pre_nms: int. Number of bboxes to keep before NMS is applied
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.stacked_convs = stacked_convs
        super(RetinaHead, self).__init__(
                self.num_classes,
                feat_channels=feat_channels,
                octave_base_scale=octave_base_scale,
                scales_per_octave=scales_per_octave,
                anchor_ratios=anchor_ratios,
                anchor_strides=anchor_strides,
                target_means=target_means,
                target_stds=target_stds
                )
        self.anchor_target = anchor_target.AnchorTarget(
            target_means=self.target_means,
            target_stds=self.target_stds,
            positive_fraction=1.0, # no sampling TODO: pass sampler as arg into anchor target generator
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr)
        #TODO make losses package common to all models
        self.class_loss = functools.partial(losses.focal_loss, alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
        self.bbox_loss = losses.retinanet_bbox_loss
        # Retina head being single stage head is the final box predictor stage, so it needs NMS specific parameters
        self.num_pre_nms = num_pre_nms
        self.nms_threshold = nms_threshold
        self.min_confidence = min_confidence
        self.max_instances = max_instances
        self.soft_nms_sigma = soft_nms_sigma


    def _init_layers(self):
        self.cls_convs = []
        self.cls_conv_bns = []
        self.reg_convs = []
        self.reg_conv_bns = []

        def bias_init_with_prob(prior_prob):
            """ initialize conv/fc bias value according to giving probablity"""
            bias_init = float(-np.log((1 - prior_prob) / prior_prob))
            return bias_init

        for i in range(self.stacked_convs):
            self.cls_convs.append(
                    layers.Conv2D(self.feat_channels, (3, 3), padding='same',
                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                        activation=None, name='cls_conv_{}'.format(i+1)))
            self.cls_conv_bns.append(layers.BatchNormalization(axis=-1, momentum=0.997, epsilon=1e-4, name='cls_conv_bn_{}'.format(i+1)))
            self.reg_convs.append(
                    layers.Conv2D(self.feat_channels, (3, 3), padding='same',
                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                        activation=None, name='reg_conv_{}'.format(i+1)))
            self.reg_conv_bns.append(layers.BatchNormalization(axis=-1, momentum=0.997, epsilon=1e-4, name='reg_conv_bn_{}'.format(i+1)))

        self.retina_cls = layers.Conv2D(self.num_anchors * self.num_classes, (3, 3),
                            padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                            bias_initializer=tf.constant_initializer(value=bias_init_with_prob(0.01)),
                            name='retina_cls')
        self.retina_reg = layers.Conv2D(self.num_anchors * 4, (3, 3), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                            name='retina_reg')


    @tf.function(experimental_relax_shapes=True)
    def forward_single(self, feat, is_training=False):
        cls_feat = feat
        reg_feat = feat
        for cls_conv, cls_conv_bn in zip(self.cls_convs, self.cls_conv_bns):
            cls_feat = cls_conv(cls_feat)
            cls_feat = cls_conv_bn(cls_feat, training=is_training)
            cls_feat = layers.Activation('relu')(cls_feat)
        for reg_conv, reg_conv_bn in zip(self.reg_convs, self.reg_conv_bns):
            reg_feat = reg_conv(reg_feat)
            reg_feat = reg_conv_bn(reg_feat, training=is_training)
            reg_feat = layers.Activation('relu')(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred


    @tf.function(experimental_relax_shapes=True)
    def call(self, feats, is_training=False):
        """
        Args
        ---
            feats (list[Tensor]): list of [batch_size, feat_map_height, feat_map_width, channels]
        Returns
        ---
        """
        layer_outputs = []
        for feat in feats: # for every anchors feature maps
            scores, deltas = self.forward_single(feat)
            layer_outputs.append([scores, deltas])
        outputs = list(zip(*layer_outputs))
        cls_scores_list, deltas_list = outputs
        return cls_scores_list, deltas_list


    def loss(self, cls_scores, pred_deltas, gt_bboxes, gt_labels, img_metas):
        """
            Args:
                cls_scores (list): scores per level, each entry in list is (N, feat_h, feat_w, num_anchors*num_classes)
                pred_deltas (list): deltas per level
                gt_bboxes (list): list of GT bboxes for images in batch
                gt_labels (list): list of GT labels for images in batch
                img_metas (list):
            Returns:
                Scalars class loss and box loss
        """
        num_images = len(img_metas)
        num_levels = len(cls_scores)

        featmap_sizes = []
        for level in range(num_levels):
            scores = cls_scores[level]
            scores_shape = tf.shape(scores)
            featmap_sizes.append([scores_shape[1], scores_shape[2]])

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas)
        num_anchors_by_level = [tf.shape(anchor)[0] for anchor in anchor_list[0]]
        (target_matches_list, target_deltas_list, inside_weights_list, 
                outside_weights_list, num_pos, _) = self.anchor_target.get_targets(anchor_list, 
                        valid_flag_list, gt_bboxes, img_metas, gt_labels)

        BG_CLASS = 0
        level_targets_list = []
        level_deltas_list = []

        # compute loss by level
        cls_losses = []
        bbox_losses = []

        prev_num_anchors = 0
        for level in range(num_levels):
            tmp_matches_list = []
            tmp_deltas_list = []
            anchor_start_idx = prev_num_anchors
            anchor_end_idx = anchor_start_idx + num_anchors_by_level[level]
            for img_idx in range(num_images):
                img_matches_for_level = target_matches_list[img_idx][anchor_start_idx:anchor_end_idx]
                img_deltas_for_level = target_deltas_list[img_idx][anchor_start_idx:anchor_end_idx]
                tmp_matches_list.append(tf.expand_dims(img_matches_for_level, axis=0))
                tmp_deltas_list.append(tf.expand_dims(img_deltas_for_level, axis=0))
            level_targets = tf.concat(tmp_matches_list, axis=0)
            level_deltas = tf.concat(tmp_deltas_list, axis=0)
            # compute loss
            selected = tf.where(level_targets >= BG_CLASS)
            scores_selected = tf.gather_nd(tf.reshape(cls_scores[level], [num_images, -1, self.num_classes]), selected)
            labels_selected = tf.gather_nd(level_targets, selected)
            class_loss = self.class_loss(scores_selected, labels_selected, avg_factor=num_pos / num_images)
            cls_losses.append(class_loss)
            selected = tf.where(level_targets > BG_CLASS)
            deltas_selected = tf.gather_nd(tf.reshape(pred_deltas[level], [num_images, -1, 4]), selected)
            target_deltas_selected = tf.gather_nd(level_deltas, selected)
            # for AMP
            deltas_selected = tf.cast(deltas_selected, tf.float32)
            target_deltas_selected = tf.cast(target_deltas_selected, tf.float32)
            bbox_loss = self.bbox_loss(deltas_selected, target_deltas_selected, avg_factor=num_pos / num_images)
            bbox_losses.append(bbox_loss)
            prev_num_anchors = anchor_end_idx

        return tf.add_n(cls_losses), tf.add_n(bbox_losses)


    @tf.function(experimental_relax_shapes=True)
    def _get_proposals_single(self,
                           scores_list,
                           deltas_list,
                           anchors_list,
                           img_shape,
                           with_probs,
                           training=True):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(scores_list) == len(deltas_list) == len(anchors_list)
        level_ids = []
        mlvl_deltas = []
        mlvl_scores = []
        mlvl_anchors = []
        mlvl_proposals = []
        num_levels = len(scores_list)
        for idx in range(num_levels):
            probs = tf.keras.layers.Activation(tf.nn.sigmoid, dtype=tf.float32)(scores_list[idx])
            deltas = deltas_list[idx]
            anchors = anchors_list[idx]
            probs = tf.reshape(probs, [-1, self.num_classes])
            deltas = tf.reshape(deltas, [-1, 4])
            pre_nms_limit = tf.math.minimum(self.num_pre_nms, tf.shape(anchors)[0])
            max_probs = tf.reduce_max(probs, axis=1)
            ix = tf.nn.top_k(max_probs, k=pre_nms_limit).indices # top k for each level (as per paper)
            level_anchors = tf.gather(anchors, ix)
            level_deltas = tf.gather(deltas, ix)
            level_scores = tf.gather(probs, ix) # these contain max_probs
            mlvl_deltas.append(level_deltas)
            mlvl_scores.append(level_scores)
            mlvl_anchors.append(level_anchors)
        scores = tf.concat(mlvl_scores, axis=0)
        anchors = tf.concat(mlvl_anchors, axis=0)
        deltas = tf.concat(mlvl_deltas, axis=0)
        proposals = transforms.delta2bbox(anchors, deltas, self.target_means, self.target_stds)
        # Clip to valid area
        window = tf.stack([0., 0., img_shape[0], img_shape[1]])
        proposals = transforms.bbox_clip(proposals, window)
        return self.batched_nms(proposals, scores, self.max_instances, self.nms_threshold)


    def batched_nms(self, bboxes, scores, max_out_count, nms_threshold):
        """
        TODO: move to utils
        Args:
            bboxes: (N, 4)
            scores: (N, num_classes)
        In order to perform NMS independently per class, we add an offset to all
        the boxes. The offset is dependent only on the class idx, and is large
        enough so that boxes from different classes do not overlap.
        """
        # repeat boxes to match scores shape
        bboxes = tf.repeat(tf.expand_dims(bboxes, axis=1), self.num_classes, axis=1)
        # filter low conf boxes
        mask = tf.where(tf.math.greater_equal(scores, self.min_confidence))
        bboxes = tf.gather_nd(bboxes, mask)
        scores = tf.gather_nd(scores, mask)
        labels = mask[:, 1] + 1 # 0 is BG

        if tf.shape(bboxes)[0] == 0:
            return tf.zeros([0, 4], dtype=tf.float32), tf.zeros([0], dtype=tf.int64), tf.zeros([0], dtype=tf.float32)

        max_coordinate = tf.reduce_max(bboxes)
        offsets = tf.cast(labels, bboxes.dtype) * (max_coordinate + 1)
        bboxes_for_nms = bboxes + offsets[:, None]
        keep, selected_scores, _ = tf.raw_ops.NonMaxSuppressionV5(boxes=bboxes_for_nms,
                scores=scores, max_output_size=max_out_count,
                iou_threshold=nms_threshold, score_threshold=0.0, 
                soft_nms_sigma=self.soft_nms_sigma)
        bboxes = tf.gather(bboxes, keep)
        labels = tf.gather(labels, keep)
        return bboxes, labels, selected_scores

