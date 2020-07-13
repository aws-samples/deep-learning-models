from .. import builder
from ..registry import HEADS
from awsdet.core.bbox import bbox_target
import tensorflow as tf


@HEADS.register_module
class CascadeHead(tf.keras.Model): 
    def __init__(self, 
                 num_stages=3,
                 stage_loss_weights=[1, 0.5, 0.25],
                 iou_thresholds=[0.5, 0.6, 0.7],
                 reg_class_agnostic=True,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 **kwargs):
        super(CascadeHead, self).__init__(**kwargs)

        assert len(stage_loss_weights) == num_stages
        assert len(bbox_head) == num_stages
        assert len(iou_thresholds) == num_stages

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights

        if bbox_roi_extractor is not None and bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
            self.bbox_heads = [builder.build_head(head) for head in bbox_head]
            self.reg_class_agnostic = reg_class_agnostic # used for build targets
            self.bbox_targets = []
            for iou, bbox_head in zip(iou_thresholds, self.bbox_heads):
                target = bbox_target.ProposalTarget(
                    target_means=bbox_head.target_means,
                    target_stds=bbox_head.target_stds, 
                    num_rcnn_deltas=512,
                    positive_fraction=0.25,
                    pos_iou_thr=iou,
                    neg_iou_thr=0.1,
                    reg_class_agnostic=self.reg_class_agnostic)
                self.bbox_targets.append(target)
    
    def call(self, 
             proposals_list, 
             gt_boxes, 
             gt_class_ids, 
             img_metas, 
             rcnn_feature_maps, 
             training=True):
        '''
        Args
        ---
            proposals_list: List of Tensors of shape [num_proposals, (ymin, xmin, ymax, xmax)]
                num_proposals=levels * proposals per level. levels refer to FPN levels. 
                Length of list is the batch size
            gt_boxes: Tensor of shape [batch_size, 4]
            gt_class_ids: Tensor of shape [batch_size]
            img_metas: Tensor of shape [11]
            rcnn_feature_maps: List of outputs from the FPN

        Returns
        ---
            logits: List of Tensors of shape [num_rois, num_classes]
            probs = List of Tensors of shape [num_rois, num_classes] 
            deltas = List of Tensors of shape [num_rois, (dy, dx, log(dh), log(dw))]
                Note: this head is class agnostic so only 1 delta per roi. 
                Different from normal BBox used for Faster RCNN.
            target_matches = List of Tensors of shape [num_rois]
            target_deltas = List of Tensors of shape [num_rois, (dy, dx, log(dh), log(dw))]
            in_weights = List of Tensors of shape [num_rois]
            out_weights = List of Tensors of shape [num_rois]
            rois_list = Tensor of shape [num_rois, (ymin, xmin, ymax, xmax)]
                Only need the final one for detection

            All List lengths are == num_stages
        '''
        logits = []
        probs = []
        deltas = []
        target_matches = []
        target_deltas = []
        in_weights = []
        out_weights = []
        for bbox_head, bbox_target in zip(self.bbox_heads, self.bbox_targets):
            if training: # get target value for these proposal target label and target delta
                rois_list, rcnn_target_matches, rcnn_target_deltas, inside_weights, outside_weights = bbox_target.build_targets(
                    proposals_list, gt_boxes, gt_class_ids, img_metas)
                target_matches.append(rcnn_target_matches)
                target_deltas.append(rcnn_target_deltas)
                in_weights.append(inside_weights)
                out_weights.append(outside_weights)
            else:
                rois_list = proposals_list

            rcnn_class_logits, rcnn_probs, rcnn_deltas  = self._forward_step(bbox_head,
                                                                                rois_list,
                                                                                img_metas,
                                                                                rcnn_feature_maps,
                                                                                training=training)

            logits.append(rcnn_class_logits)
            probs.append(rcnn_probs)
            deltas.append(rcnn_deltas)

        '''
        # apply rcnn deltas to bboxes and use them as new proposals for next stage
        # TODO figure out better way to apply all deltas since get bboxes will only 
        # work with a single image. 
        batch_size = len(proposals_list)
        roi_size = tf.cast(proposals_list[0].shape[0] / batch_size, tf.int32) # all rois get padded to max instances
        num_classes = rcnn_class_logits.shape[1]
        rcnn_probs = tf.cast(rcnn_probs, tf.float32)
        rcnn_deltas = tf.cast(rcnn_deltas, tf.float32)
        reshaped_probs = tf.reshape(rcnn_probs, [batch_size, roi_size, num_classes])
        if bbox_head.reg_class_agnostic:
            reshaped_deltas = tf.reshape(rcnn_deltas, [batch_size, roi_size, 4])
        else:
            reshaped_deltas = tf.reshape(rcnn_deltas, [batch_size, roi_size, num_classes * 4])
        proposals_list = []
        for i in range(batch_size):
            detections_list = bbox_head.get_bboxes(reshaped_probs[i], reshaped_deltas[i], rois_list[i], tf.expand_dims(img_metas[i], 0))
            proposals_list.append(detections_list[0][0])

        print(proposals_list)
        '''

        detections_list = bbox_head.get_bboxes(rcnn_probs, rcnn_deltas, rois_list, img_metas)
        proposals_list = []
        for i in range(len(detections_list)):
            proposals_list.append(detections_list[i][0])

        return logits, probs, deltas, target_matches, target_deltas, in_weights, out_weights, rois_list
    
    @tf.function(experimental_relax_shapes=True)
    def _forward_step(self, 
                      bbox_head, 
                      rois_list,
                      img_metas, 
                      rcnn_feature_maps,
                      training=True):
        pooled_regions_list = self.bbox_roi_extractor(
            (rois_list, rcnn_feature_maps, img_metas), training=training)

        rcnn_class_logits, rcnn_probs, rcnn_deltas = bbox_head(pooled_regions_list, training=training)

        return rcnn_class_logits, rcnn_probs, rcnn_deltas
