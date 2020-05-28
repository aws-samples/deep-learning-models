# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
import functools
from .anchor_head import AnchorHead
from awsdet.core.bbox import transforms
from awsdet.models.utils.misc import calc_pad_shapes
from awsdet.core.anchor import anchor_target
from awsdet.models.losses import losses
from ..registry import HEADS


@HEADS.register_module
class RPNHead(AnchorHead):

    def __init__(self,
                 anchor_scales,
                 anchor_ratios,
                 anchor_strides,
                 nms_threshold=0.7,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_samples=256,
                 feat_channels=512,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 weight_decay=1e-4,
                 num_pre_nms_train=12000,
                 num_post_nms_train=2000,
                 num_pre_nms_test=6000,
                 num_post_nms_test=300):
        '''
        Network head of Region Proposal Network.

                                      |-- > rpn_cls (1x1 conv)
        input - rpn_conv (3x3 conv) --
                                      |-- > rpn_reg (1x1 conv)

        Attributes
        ---
            anchor_scales: 1D array of anchor sizes in pixels.
            anchor_ratios: 1D array of anchor ratios of width/height.
            anchor_strides: Stride of the feature map relative 
                to the image in pixels.
            proposal_count: int. RPN proposals kept after non-maximum 
                suppression.
            nms_threshold: float. Non-maximum suppression threshold to 
                filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_samples: int.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        self.num_classes = 2
        self.weight_decay = weight_decay
        super(RPNHead, self).__init__(
                self.num_classes,
                feat_channels=feat_channels,
                anchor_scales=anchor_scales,
                anchor_ratios=anchor_ratios,
                anchor_strides=anchor_strides,
                target_means=target_means,
                target_stds=target_stds
                )
        self.num_samples = num_samples
        self.nms_threshold = nms_threshold
        self.num_pre_nms_train = num_pre_nms_train
        self.num_post_nms_train = num_post_nms_train
        self.num_pre_nms_test = num_pre_nms_test
        self.num_post_nms_test = num_post_nms_test

        self.anchor_target = anchor_target.AnchorTarget(
            target_means=self.target_means,
            target_stds=self.target_stds,
            num_samples=num_samples,
            positive_fraction=positive_fraction,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr)

        self.rpn_class_loss = losses.rpn_class_loss
        self.rpn_bbox_loss = losses.rpn_bbox_loss


    def _init_layers(self):
        # Shared convolutional base of the RPN
        self.rpn_conv_shared = layers.Conv2D(self.feat_channels, (3, 3), padding='same',
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                             kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                             activation='relu',
                                             name='rpn_conv_shared')
        self.rpn_class_raw = layers.Conv2D(self.num_anchors * self.num_classes, (1, 1),
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                           kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                           name='rpn_class_raw')
        self.rpn_delta_pred = layers.Conv2D(self.num_anchors * 4, (1, 1),
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                                           kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                           name='rpn_bbox_pred')


    @tf.function(experimental_relax_shapes=True)
    def forward_single(self, feat):
        shared = self.rpn_conv_shared(feat)
        x = self.rpn_class_raw(shared)
        rpn_class_logits = layers.Activation('linear', dtype=tf.float32)(x) # for AMP
        rpn_probs = layers.Activation('softmax', dtype='float32', name='rpn_probs')(rpn_class_logits)
        x = self.rpn_delta_pred(shared)
        rpn_deltas = layers.Activation('linear', dtype=tf.float32)(x)
        return rpn_class_logits, rpn_probs, rpn_deltas


    @tf.function(experimental_relax_shapes=True)
    def call(self, feats):
        """
        Args
        ---
            feats (list[Tensor]): list of [batch_size, feat_map_height, feat_map_width, channels]
        Returns
        ---
            rpn_class_logits: [batch_size, num_anchors, 2]
            rpn_probs: [batch_size, num_anchors, 2]
            rpn_deltas: [batch_size, num_anchors, 4]
        """
        layer_outputs = []
        for feat in feats: # for every anchors feature maps
            rpn_class_logits, rpn_probs, rpn_deltas = self.forward_single(feat)
            layer_outputs.append([rpn_class_logits, rpn_probs, rpn_deltas])
        outputs = list(zip(*layer_outputs))
        rpn_class_logits, rpn_probs, rpn_deltas = outputs
        return rpn_class_logits, rpn_probs, rpn_deltas


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
        featmap_sizes = []
        cls_logits_list = [[] for _ in range(num_images)]
        deltas_list = [[] for _ in range(num_images)]
        for scores, deltas in zip(cls_scores, pred_deltas):
            featmap_shape = tf.shape(scores)
            featmap_sizes.append((featmap_shape[1], featmap_shape[2]))
            img_level_scores = tf.split(scores, num_images, axis=0)
            img_level_deltas = tf.split(deltas, num_images, axis=0)
            level_scores = [tf.reshape(level_score, [-1, self.num_classes]) for level_score in img_level_scores]
            level_deltas = [tf.reshape(level_delta, [-1, 4]) for level_delta in img_level_deltas]
            for img_id, (level_score, level_delta) in enumerate(zip(level_scores, level_deltas)):
                cls_logits_list[img_id].append(level_score)
                deltas_list[img_id].append(level_delta)
        cls_logits_list = [tf.concat(img_scores, axis=0) for img_scores in cls_logits_list]
        deltas_list = [tf.concat(delta, axis=0) for delta in deltas_list]
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas)
        (target_matches_list, target_deltas_list, inside_weights_list,
                outside_weights_list, num_pos, num_neg) = self.anchor_target.get_targets(
                        anchor_list, valid_flag_list, gt_bboxes, img_metas,
                        gt_labels_list=None) # use labels (0 1)
        total_samples = float(num_pos + num_neg)
        target_matches = tf.concat(target_matches_list, axis=0)
        target_deltas = tf.concat(target_deltas_list, axis=0)
        inside_weights = tf.concat(inside_weights_list, axis=0)
        outside_weights = tf.concat(outside_weights_list, axis=0)
        selected = tf.where(target_matches >= 0)[:, 0]
        cls_logits = tf.concat(cls_logits_list, axis=0)
        scores_selected = tf.gather(cls_logits, selected)
        labels_selected = tf.gather(target_matches, selected)
        class_loss = self.rpn_class_loss(scores_selected, labels_selected,
                        avg_factor=total_samples / num_images)
        deltas = tf.concat(deltas_list, axis=0)
        bbox_loss = self.rpn_bbox_loss(deltas,
                                       target_deltas, 
                                       inside_weights,
                                       outside_weights)
        return class_loss, bbox_loss 


    @tf.function(experimental_relax_shapes=True)
    def _get_proposals_single(self,
                              rpn_probs,
                              rpn_deltas,
                              mlvl_anchors,
                              img_shape,
                              with_probs,
                              training=True):
        """
        Calculate proposals per image
        Args:
        Returns:
        """
        if training:
            num_pre_nms=self.num_pre_nms_train
            proposal_count = self.num_post_nms_train
        else:
            num_pre_nms=self.num_pre_nms_test
            proposal_count = self.num_post_nms_test

        level_ids = []
        mlvl_scores = []
        mlvl_deltas = []
        mlvl_valid_anchors = []

        mlvl_proposals = []
        num_levels = len(rpn_probs)
        for idx in range(num_levels):
            level_probs = tf.reshape(rpn_probs[idx], [-1, 2]) # H, W, probs -> H * W, probs
            level_scores = level_probs[:, 1]
            level_deltas = tf.reshape(rpn_deltas[idx], [-1, 4])
            level_anchors = mlvl_anchors[idx]
            pre_nms_limit = tf.math.minimum(num_pre_nms, tf.shape(level_anchors)[0])
            ix = tf.nn.top_k(level_scores, pre_nms_limit, sorted=False).indices
            level_scores = tf.gather(level_scores, ix)
            level_deltas = tf.gather(level_deltas, ix)
            level_anchors = tf.gather(level_anchors, ix)
            mlvl_scores.append(level_scores)
            mlvl_deltas.append(level_deltas)
            mlvl_valid_anchors.append(level_anchors)
            level_ids.append(tf.fill([tf.shape(level_scores)[0],], idx))
        scores = tf.concat(mlvl_scores, axis=0)
        anchors = tf.concat(mlvl_valid_anchors, axis=0)
        deltas = tf.concat(mlvl_deltas, axis=0)

        # get refined anchors
        proposals = transforms.delta2bbox(anchors, deltas,self.target_means, self.target_stds)
        # Clip to valid area
        window = tf.stack([0., 0., img_shape[0], img_shape[1]])
        proposals = transforms.bbox_clip(proposals, window)
        ids = tf.concat(level_ids, axis=0)
        # NMS is appied per level independent of others
        keep = self.batched_nms(proposals, scores, ids, proposal_count, self.nms_threshold)
        proposals = tf.gather(proposals, keep)
        return tf.stop_gradient(proposals)


    def batched_nms(self, bboxes, scores, inds, max_out_count, nms_threshold):
        """
        TODO: move to utils
        Args:
            bboxes: (N, 4)
            scores: (N,)
            inds: (N,) indicates the class/level of bbox
        In order to perform NMS independently per class, we add an offset to all
        the boxes. The offset is dependent only on the class idx, and is large
        enough so that boxes from different classes do not overlap.
        """
        max_coordinate = tf.reduce_max(bboxes)
        offsets = tf.cast(inds, bboxes.dtype) * (max_coordinate + 1)
        bboxes_for_nms = bboxes + offsets[:, None]
        return tf.raw_ops.NonMaxSuppressionV2(boxes=bboxes_for_nms,
                scores=scores, max_output_size=max_out_count, iou_threshold=nms_threshold)
        
