# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
import functools
from awsdet.core.bbox import transforms
from awsdet.models.utils.misc import calc_pad_shapes

from awsdet.core.anchor import anchor_generator, anchor_target
from awsdet.models.losses import losses
from ..registry import HEADS

@HEADS.register_module
class RPNHead(tf.keras.Model):

    def __init__(self,
                 anchor_scales=(32, 64, 128, 256, 512),
                 anchor_ratios=(0.5, 1, 2),
                 anchor_feature_strides=(4, 8, 16, 32, 64),
                 nms_threshold=0.7,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 weight_decay=1e-4,
                 num_pre_nms_train=12000,
                 num_post_nms_train=2000,
                 num_pre_nms_test=6000,
                 num_post_nms_test=300,
                 padded_img_shape=(1024, 1024),
                 **kwargs):
        '''
        Network head of Region Proposal Network.

                                      |-- > rpn_cls (1x1 conv)
        input - rpn_conv (3x3 conv) --
                                      |-- > rpn_reg (1x1 conv)

        Attributes
        ---
            anchor_scales: 1D array of anchor sizes in pixels.
            anchor_ratios: 1D array of anchor ratios of width/height.
            anchor_feature_strides: Stride of the feature map relative 
                to the image in pixels.
            proposal_count: int. RPN proposals kept after non-maximum 
                suppression.
            nms_threshold: float. Non-maximum suppression threshold to 
                filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_rpn_deltas: int.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        super(RPNHead, self).__init__(**kwargs)
        self.num_rpn_deltas = num_rpn_deltas
        self.nms_threshold = nms_threshold
        self.num_pre_nms_train = num_pre_nms_train
        self.num_post_nms_train = num_post_nms_train
        self.num_pre_nms_test = num_pre_nms_test
        self.num_post_nms_test = num_post_nms_test
        self.target_means = target_means
        self.target_stds = target_stds
        self.generator = anchor_generator.AnchorGenerator(
            scales=anchor_scales,
            ratios=anchor_ratios,
            feature_strides=anchor_feature_strides,
            padding_size=padded_img_shape)
        self.anchor_target = anchor_target.AnchorTarget(
            target_means=target_means,
            target_stds=target_stds,
            num_rpn_deltas=num_rpn_deltas,
            positive_fraction=positive_fraction,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr)

        self.rpn_class_loss = losses.rpn_class_loss
        self.rpn_bbox_loss = losses.rpn_bbox_loss
        # Shared convolutional base of the RPN
        self.rpn_conv_shared = layers.Conv2D(512, (3, 3), padding='same',
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             activation='relu',
                                             name='rpn_conv_shared')
        num_anchors = len(anchor_ratios)
        self.rpn_class_raw = layers.Conv2D(num_anchors * 2, (1, 1),
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                           name='rpn_class_raw')
        self.rpn_delta_pred = layers.Conv2D(num_anchors * 4, (1, 1),
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                           name='rpn_bbox_pred')


    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True):
        '''
        Args
        ---
            inputs: [batch_size, feat_map_height, feat_map_width, channels] 
                one level of pyramid feat-maps.
        
        Returns
        ---
            rpn_class_logits: [batch_size, num_anchors, 2]
            rpn_probs: [batch_size, num_anchors, 2]
            rpn_deltas: [batch_size, num_anchors, 4]
        '''
        layer_outputs = []
        for feat in inputs: # for every anchors feature maps
            shared = self.rpn_conv_shared(feat)
            x = self.rpn_class_raw(shared)
            rpn_class_logits = layers.Activation('linear', dtype='float32')(x) # for AMP
            rpn_class_logits = tf.reshape(rpn_class_logits, [tf.shape(rpn_class_logits)[0], -1, 2])
            rpn_probs = layers.Activation('softmax', dtype='float32', name='rpn_probs')(rpn_class_logits)
            x = self.rpn_delta_pred(shared)
            rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])
            rpn_deltas = layers.Activation('linear', dtype='float32')(rpn_deltas)
            layer_outputs.append([rpn_class_logits, rpn_probs, rpn_deltas])
        outputs = list(zip(*layer_outputs))
        outputs = [tf.concat(list(o), axis=1) for o in outputs]
        rpn_class_logits, rpn_probs, rpn_deltas = outputs
        return rpn_class_logits, rpn_probs, rpn_deltas


    @tf.function(experimental_relax_shapes=True)
    def loss(self, inputs):
        """
        :param rpn_class_logits: [N, 2]
        :param rpn_deltas: [N, 4]
        :param gt_boxes:  [GT_N]
        :param gt_class_ids:  [GT_N]
        :param img_metas: [11]
        :return:
        """
        rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas = inputs
        # valid_flags indicates anchors located in padded area or not.
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)
        # combine batches for loss calculation
        rpn_class_logits = tf.reshape(rpn_class_logits, ([rpn_class_logits.shape[0] * rpn_class_logits.shape[1], -1]))
        rpn_deltas = tf.reshape(rpn_deltas, ([rpn_deltas.shape[0] * rpn_deltas.shape[1], -1]))
        rpn_target_matches, rpn_target_deltas, rpn_inside_weights, rpn_outside_weights = self.anchor_target.build_targets(anchors, valid_flags, gt_boxes, gt_class_ids)
        rpn_selected = tf.where(rpn_target_matches >= 0)[:, 0]
        rpn_score_selected = tf.gather(rpn_class_logits, rpn_selected)
        rpn_labels_selected = tf.gather(rpn_target_matches, rpn_selected)
        rpn_class_loss = self.rpn_class_loss(rpn_score_selected, rpn_labels_selected, rpn_deltas=self.num_rpn_deltas)
        rpn_bbox_loss = self.rpn_bbox_loss(rpn_deltas,
                                           rpn_target_deltas, 
                                           rpn_inside_weights,
                                           rpn_outside_weights)
        return rpn_class_loss, rpn_bbox_loss


    def get_proposals(self,
                      rpn_probs,
                      rpn_deltas,
                      img_metas,
                      with_probs=False,
                      training=True):
        '''
        Calculate proposals.
        
        Args
        ---
            rpn_probs: [batch_size, num_anchors, (bg prob, fg prob)]
            rpn_deltas: [batch_size, num_anchors, (dy, dx, log(dh), log(dw))]
            img_metas: [batch_size, 11]
            with_probs: bool.
        
        Returns
        ---
            proposals_list: list of [num_proposals, (y1, x1, y2, x2)] in 
                normalized coordinates if with_probs is False. 
                Otherwise, the shape of proposals in proposals_list is 
                [num_proposals, (y1, x1, y2, x2, score)]
        
        Note that num_proposals is no more than proposal_count. And different 
           images in one batch may have different num_proposals.
        '''
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)
        rpn_probs = rpn_probs[:, :, 1]
        pad_shapes = calc_pad_shapes(img_metas)
        batch_size = img_metas.shape[0]
        proposals_list = [
            self._get_proposals_single(
                rpn_probs[i], rpn_deltas[i], anchors, valid_flags[i], pad_shapes[i], with_probs, training)
            for i in range(batch_size)
        ]
        return proposals_list


    @tf.function(experimental_relax_shapes=True)
    def _get_proposals_single(self,
                              rpn_probs,
                              rpn_deltas,
                              anchors,
                              valid_flags,
                              img_shape,
                              with_probs,
                              training=True):
        '''
        Calculate proposals.
        
        Args
        ---
            rpn_probs: [num_anchors]
            rpn_deltas: [num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [num_anchors, (y1, x1, y2, x2)] anchors defined in 
                pixel coordinates.
            valid_flags: [num_anchors]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            with_probs: bool.
        
        Returns
        ---
            proposals: [num_proposals, (y1, x1, y2, x2)] in normalized 
                coordinates.
        '''
        H = img_shape[0]
        W = img_shape[1]
        # filter invalid anchors, int => bool
        valid_flags = tf.cast(valid_flags, tf.bool)
        rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
        rpn_deltas = tf.boolean_mask(rpn_deltas, valid_flags)
        anchors = tf.boolean_mask(anchors, valid_flags)
        # Improve performance
        if training:
            num_pre_nms=self.num_pre_nms_train
        else:
            num_pre_nms=self.num_pre_nms_test
        pre_nms_limit = tf.math.minimum(num_pre_nms, tf.shape(anchors)[0])
        ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=False).indices
        rpn_probs = tf.gather(rpn_probs, ix)
        rpn_deltas = tf.gather(rpn_deltas, ix)
        anchors = tf.gather(anchors, ix)
        # Get refined anchors
        proposals = transforms.delta2bbox(anchors, rpn_deltas,
                                          self.target_means, self.target_stds)
        # Clip to valid area
        window = tf.stack([0., 0., H, W])
        proposals = transforms.bbox_clip(proposals, window)
        if training:
            proposal_count = self.num_post_nms_train
        else:
            proposal_count = self.num_post_nms_test
        rpn_probs = tf.cast(rpn_probs, proposals.dtype)
        # indices = tf.image.non_max_suppression(proposals, rpn_probs,
        #                                             max_output_size=proposal_count,
        #                                             iou_threshold=self.nms_threshold)
        indices = tf.raw_ops.NonMaxSuppressionV2(boxes=proposals, 
                                                    scores=rpn_probs,
                                                    max_output_size=proposal_count,
                                                    iou_threshold=self.nms_threshold)
        proposals = tf.stop_gradient(tf.gather(proposals, indices))
        if with_probs:
            proposal_probs = tf.expand_dims(tf.gather(rpn_probs, indices), axis=1)
            proposals = tf.concat([proposals, proposal_probs], axis=1)
        return proposals

