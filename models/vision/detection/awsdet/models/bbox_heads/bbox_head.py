# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers

from awsdet.core.bbox import transforms
from awsdet.models.losses import losses
from ..utils.misc import (calc_batch_padded_shape, calc_img_shapes, calc_pad_shapes)
from ..registry import HEADS


@HEADS.register_module
class BBoxHead(tf.keras.Model):
    def __init__(self, 
                 num_classes, 
                 pool_size=(7, 7), # ROI feature size
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 min_confidence=0.7,
                 nms_threshold=0.3,
                 max_instances=100,
                 num_rcnn_deltas=512,
                 weight_decay=1e-4,
                 use_conv=False,
                 use_bn=False,
                 label_smoothing=0.0,
                 soft_nms_sigma=0.0,
                 **kwags):
        super(BBoxHead, self).__init__(**kwags)
        
        self.num_classes = num_classes
        self.pool_size = tuple(pool_size)
        self.target_means = target_means
        self.target_stds = target_stds
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances
        self.num_rcnn_deltas=num_rcnn_deltas
        self.rcnn_class_loss = losses.rcnn_class_loss
        self.rcnn_bbox_loss = losses.rcnn_bbox_loss
        self.use_conv = use_conv
        self.use_bn = (use_bn and not use_conv)
        self.label_smoothing = label_smoothing
        self.soft_nms_sigma = soft_nms_sigma
        
        roi_feature_size=(7, 7, 256)
        if not use_conv:
            self._flatten_layer = layers.Flatten()
            
            self._flatten_bn = layers.BatchNormalization(scale=True, epsilon=1e-5, name='flatten_bn')
            
            self._fc1 = layers.Dense(1024, name='fc1', activation=None,
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     input_shape=[roi_feature_size]
                                     )
            self._fc1_bn = layers.BatchNormalization(name='fc1_bn')
            
            self._fc2 = layers.Dense(1024, name='fc2', activation=None,
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     )
            
            self._fc2_bn = layers.BatchNormalization(name='fc2_bn')

            self.rcnn_class_logits = layers.Dense(num_classes,
                                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                  name='rcnn_class_logits')

            self.rcnn_delta_fc = layers.Dense(num_classes * 4, 
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                              name='rcnn_bbox_fc')
        else:
            self._conv1 = layers.Conv2D(1024, (3, 3), padding='same',
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        activation='relu',
                                        name='rcnn_conv_1')
            
            self._conv2 = layers.Conv2D(1024, self.pool_size, padding='valid',
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        activation='relu',
                                        name='rcnn_conv_2')

            self.rcnn_class_logits = layers.Conv2D(num_classes, (1, 1),
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        activation='linear',
                                        name='rcnn_class_logit')
            
            self.rcnn_delta_cv = layers.Conv2D(num_classes * 4, (1, 1),
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        activation='linear',
                                        name='rcnn_delta_cv')


    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True):
        '''
        Args
        ---
            pooled_rois_list: List of [num_rois, pool_size, pool_size, channels]
        
        Returns
        ---
            rcnn_class_logits_list: List of [num_rois, num_classes]
            rcnn_probs_list: List of [num_rois, num_classes]
            rcnn_deltas_list: List of [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        '''

        pooled_rois_list = inputs
        pooled_rois = tf.concat(pooled_rois_list, axis=0)
        
        if not self.use_conv:
            x = self._flatten_layer(pooled_rois)
            if self.use_bn:
                x = self._flatten_bn(x, training=training)
            x = self._fc1(x)
            if self.use_bn:
                x = self._fc1_bn(x, training=training)
            x = layers.Activation('relu')(x)
            x = self._fc2(x)
            if self.use_bn:
                x = self._fc2_bn(x, training=training)
            x = layers.Activation('relu')(x)
            logits = self.rcnn_class_logits(x)
            logits = layers.Activation('linear', dtype='float32')(logits) # for AMP
            probs = layers.Activation('softmax', dtype='float32', name='rcnn_probs')(logits)
            deltas = self.rcnn_delta_fc(x)
            deltas = layers.Activation('linear', dtype='float32')(deltas)
            return logits, probs, deltas
        else:
            x = self._conv1(pooled_rois)
            x = self._conv2(x)
            # x = layers.AveragePooling2D(self.pool_size)(pooled_rois)
            logits = self.rcnn_class_logits(x)
            logits = layers.Activation('linear', dtype='float32')(logits) # for AMP
            # rpn_class_logits = tf.reshape(rpn_class_logits, [tf.shape(rpn_class_logits)[0], -1, 2])
            logits = tf.reshape(logits, [-1, self.num_classes])
            probs = layers.Activation('softmax', dtype='float32', name='rcnn_probs')(logits)
            deltas = self.rcnn_delta_cv(x)
            deltas = layers.Activation('linear', dtype='float32')(deltas)
            deltas = tf.reshape(deltas, [-1, self.num_classes * 4])
            return logits, probs, deltas


    @tf.function(experimental_relax_shapes=True)
    def loss(self, inputs):
        """
        :param rcnn_class_logits:
        :param rcnn_deltas:
        :param rcnn_target_matches:
        :param rcnn_target_deltas:
        :return:
        """
        rcnn_class_logits, rcnn_deltas, rcnn_target_matches, rcnn_target_deltas, inside_weights, outside_weights = inputs
        rcnn_class_loss = self.rcnn_class_loss(rcnn_class_logits, rcnn_target_matches, 
                                               avg_factor=self.num_rcnn_deltas, label_smoothing=self.label_smoothing)
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_deltas, rcnn_target_deltas, 
                                             inside_weights, outside_weights)

        return rcnn_class_loss, rcnn_bbox_loss
 

    def get_bboxes(self, rcnn_probs, rcnn_deltas, rois_list, img_metas):
        '''
        Args
        ---
            rcnn_probs_list: List of [num_rois, num_classes]
            rcnn_deltas_list: List of [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois_list: List of [num_rois, (y1, x1, y2, x2)]
            img_meta_list: [batch_size, 11]
        
        Returns
        ---
            detections_list: List of [num_detections, (y1, x1, y2, x2, class_id, score)]
                coordinates are in pixel coordinates.
        '''
        num_rois_list = [tf.shape(rois)[0] for rois in rois_list]
        rcnn_probs_list = tf.split(rcnn_probs, num_rois_list, 0)
        rcnn_deltas_list = tf.split(rcnn_deltas, num_rois_list, 0)
        pad_shapes = calc_pad_shapes(img_metas)
        detections_list = [self._get_bboxes_single(rcnn_probs_list[i], rcnn_deltas_list[i],
            rois_list[i], pad_shapes[i]) for i in range(img_metas.shape[0])]
        return detections_list 


    @tf.function
    def _get_bboxes_single(self, rcnn_probs, rcnn_deltas, rois, img_shape):
        '''
        Args
        ---
            rcnn_probs: [num_rois, num_classes]
            rcnn_deltas: [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois: [num_rois, (y1, x1, y2, x2)]
            img_shape: np.ndarray. [2]. (img_height, img_width)       
        '''
        H = img_shape[0] 
        W = img_shape[1] 
        
        res_scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=True)
        res_bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=True)
        res_cls = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=True)
        for cls_id in range(1, self.num_classes):
            inds = tf.where(rcnn_probs[:, cls_id] > self.min_confidence)[:, 0]
            cls_score = tf.gather(rcnn_probs[:, cls_id], inds)
            rcnn_deltas = tf.reshape(rcnn_deltas, [-1, self.num_classes, 4])
            final_bboxes = transforms.delta2bbox(tf.gather(rois, inds),
                                                tf.gather(rcnn_deltas[:, cls_id, :], inds),
                                                self.target_means, self.target_stds)
            window = tf.stack([tf.constant(0., H.dtype), tf.constant(0., W.dtype), H, W])
            final_bboxes = transforms.bbox_clip(final_bboxes, window)
            cls_score = tf.cast(cls_score, final_bboxes.dtype)
            #keep = tf.image.non_max_suppression(final_bboxes, cls_score,
            #                                    self.max_instances,
            #                                    iou_threshold=self.nms_threshold)
            keep, selected_cls_scores, _ = tf.raw_ops.NonMaxSuppressionV5 (
                                                    boxes=final_bboxes, scores=cls_score,
                                                    max_output_size=self.max_instances,
                                                    iou_threshold=self.nms_threshold,
                                                    score_threshold=0.0,
                                                    soft_nms_sigma=self.soft_nms_sigma)
            pad_size = self.max_instances - tf.size(keep)
            padded_scores = tf.pad(selected_cls_scores, paddings=[[0, pad_size]], constant_values=0.0)
            res_scores = res_scores.write(cls_id-1, padded_scores)#.mark_used()
            padded_bboxes = tf.pad(tf.gather(final_bboxes, keep), paddings=[[0, pad_size], [0, 0]], constant_values=0.0)
            res_bboxes = res_bboxes.write(cls_id-1, padded_bboxes)#.mark_used()
            padded_cls = tf.pad(tf.ones_like(keep, dtype=tf.int32) * cls_id, paddings=[[0, pad_size]], constant_values=-1)
            res_cls = res_cls.write(cls_id-1, padded_cls)#.mark_used()

        res_scores = res_scores.stack()
        res_bboxes = res_bboxes.stack()
        res_cls = res_cls.stack()

        scores_after_nms = tf.reshape(res_scores, [-1])
        bboxes_after_nms = tf.reshape(res_bboxes, [-1, 4])
        cls_after_nms = tf.reshape(res_cls, [-1])
 
        _, final_idx = tf.nn.top_k(scores_after_nms,
                                   k=tf.minimum(self.max_instances, tf.size(scores_after_nms)),
                                   sorted=False)
 
        return (tf.gather(bboxes_after_nms, final_idx),
                tf.gather(cls_after_nms, final_idx),
                tf.gather(scores_after_nms, final_idx))
