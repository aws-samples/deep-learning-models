from .. import builder
from ..registry import HEADS
from awsdet.core.bbox import bbox_target, transforms
from awsdet.models.losses import losses
import tensorflow as tf



@HEADS.register_module
class CascadeHead(tf.keras.Model): 
    def __init__(self, 
                 num_stages=3,
                 stage_loss_weights=[1, 0.5, 0.25],
                 iou_thresholds=[0.5, 0.6, 0.7],
                 reg_class_agnostic=True,
                 num_classes=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 **kwargs):
        super(CascadeHead, self).__init__(**kwargs)

        assert len(stage_loss_weights) == num_stages
        assert len(bbox_head) == num_stages
        assert len(iou_thresholds) == num_stages
        assert reg_class_agnostic or num_classes

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights

        if bbox_roi_extractor is not None and bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
            self.bbox_heads = [builder.build_head(head) for head in bbox_head]
            self.reg_class_agnostic = reg_class_agnostic # used for build targets
            self.num_classes = num_classes
            self.bbox_targets = []
            for iou, bbox_head in zip(iou_thresholds, self.bbox_heads):
                target = bbox_target.ProposalTarget(
                    target_means=bbox_head.target_means,
                    target_stds=bbox_head.target_stds, 
                    num_rcnn_deltas=512,
                    positive_fraction=0.25,
                    pos_iou_thr=iou,
                    neg_iou_thr=0.1,
                    reg_class_agnostic=self.reg_class_agnostic,
                    num_classes=1 if reg_class_agnostic else self.num_classes)
                self.bbox_targets.append(target)
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True):
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
        '''
        if training:
            proposals_list, rcnn_feature_maps, gt_boxes, \
            gt_class_ids, img_metas = inputs
        else:
            proposals_list, rcnn_feature_maps, img_metas = inputs
        batch_size = img_metas.shape[0]
        loss_dict = {}
        for i in range(self.num_stages):
            if i == 0:
                rois_list = proposals_list
            if training:
                rois_list, rcnn_target_matches, rcnn_target_deltas, inside_weights, \
                    outside_weights = self.bbox_targets[i].build_targets( \
                    rois_list, gt_boxes, gt_class_ids, img_metas)    
            pooled_regions_list = self.bbox_roi_extractor(
                (rois_list, rcnn_feature_maps, img_metas), training=training)
            rcnn_class_logits, rcnn_probs, rcnn_deltas = self.bbox_heads[i](pooled_regions_list, training=training)
            if training:
                loss_dict['rcnn_class_loss_stage_{}'.format(i)] = losses.rcnn_class_loss(rcnn_class_logits, 
                                                                                         rcnn_target_matches) * self.stage_loss_weights[i]
        
                loss_dict['rcnn_box_loss_stage_{}'.format(i)] = losses.rcnn_bbox_loss(rcnn_deltas,
                                                                                      rcnn_target_deltas, 
                                                                                      inside_weights, 
                                                                                      outside_weights) * self.stage_loss_weights[i]
            roi_shapes = [tf.shape(i)[0] for i in rois_list]
            refinements = tf.split(rcnn_deltas, roi_shapes)
            new_rois = []
            if i<(self.num_stages-1):
                for j in range(batch_size):
                    new_rois.append(tf.stop_gradient(transforms.delta2bbox(rois_list[j], refinements[j],
                                                   target_means=self.bbox_heads[i].target_means, \
                                                   target_stds=self.bbox_heads[i].target_stds)))
                rois_list = new_rois
        if training:
            return loss_dict
        else:
            detections_list = self.bbox_heads[-1].get_bboxes(rcnn_probs,
                                                            rcnn_deltas,
                                                            rois_list,
                                                            img_metas)
            detections_dict = {
                    'bboxes': detections_list[0][0],
                    'labels': detections_list[0][1],
                    'scores': detections_list[0][2]
            }
            return detections_dict

        