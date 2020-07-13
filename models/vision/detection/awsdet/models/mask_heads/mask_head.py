import tensorflow as tf
import tensorflow_addons as tfa
from awsdet.core.mask import transforms
from awsdet.models.losses import losses
from ..registry import HEADS

@HEADS.register_module
class MaskHead(tf.keras.Model):
    def __init__(self, num_classes,
                       crop_size=(28, 28),
                       weight_decay=1e-5, 
                       use_gn=False,
                       use_bn=False, 
                       loss_func=tf.nn.sigmoid_cross_entropy_with_logits):
        super().__init__()
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.crop_size = tf.constant(crop_size)
        self.loss_func = loss_func
        assert not (use_gn & use_bn), "Cannot use both group and batch norm"
        self.use_gn = use_gn
        self.use_bn = use_bn
        self._conv_0 = tf.keras.layers.Conv2D(256, (3, 3),
                                             padding="same",
                                             kernel_initializer='glorot_uniform',
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             activation=tf.keras.activations.relu,
                                             name="mask_conv_0")
        if self.use_gn:
            self._conv_0_gn = tfa.layers.GroupNormalization(groups=32)
        if self.use_bn:
            self._conv_0_bn = tf.keras.layers.BatchNormalization()
        self._conv_1 = tf.keras.layers.Conv2D(256, (3, 3),
                                             padding="same",
                                             kernel_initializer='glorot_uniform',
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             activation=tf.keras.activations.relu,
                                             name="mask_conv_1")
        if self.use_gn:
            self._conv_1_gn = tfa.layers.GroupNormalization(groups=32)
        if self.use_bn:
            self._conv_1_bn = tf.keras.layers.BatchNormalization()
        self._conv_2 = tf.keras.layers.Conv2D(256, (3, 3),
                                             padding="same",
                                             kernel_initializer='glorot_uniform',
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             activation=tf.keras.activations.relu,
                                             name="mask_conv_2")
        if self.use_gn:
            self._conv_2_gn = tfa.layers.GroupNormalization(groups=32)
        if self.use_bn:
            self._conv_2_bn = tf.keras.layers.BatchNormalization()
        self._conv_3 = tf.keras.layers.Conv2D(256, (3, 3),
                                             padding="same",
                                             kernel_initializer='glorot_uniform',
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             activation=tf.keras.activations.relu,
                                             name="mask_conv_3")
        if self.use_gn:
            self._conv_3_gn = tfa.layers.GroupNormalization(groups=32)
        if self.use_bn:
            self._conv_3_bn = tf.keras.layers.BatchNormalization()
        self._deconv = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, 
                                                    kernel_initializer='glorot_uniform',
                                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                    activation=tf.keras.activations.relu,
                                                    name="mask_deconv")
        self._masks = tf.keras.layers.Conv2D(self.num_classes, (1, 1),
                                             kernel_initializer='glorot_uniform',
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                             strides=1, name="mask_output")
        
    @tf.function(experimental_relax_shapes=True)
    def call(self, mask_rois_list, training=True):
        mask_list = []
        for mask_rois in mask_rois_list:
            mask_rois = self._conv_0(mask_rois)
            if self.use_gn:
                mask_rois = self._conv_0_gn(mask_rois)
            if self.use_bn:
                mask_rois = self._conv_0_bn(mask_rois, training=training)
            mask_rois = self._conv_1(mask_rois)
            if self.use_gn:
                mask_rois = self._conv_1_gn(mask_rois)
            if self.use_bn:
                mask_rois = self._conv_1_bn(mask_rois, training=training)
            mask_rois = self._conv_2(mask_rois)
            if self.use_gn:
                mask_rois = self._conv_2_gn(mask_rois)
            if self.use_bn:
                mask_rois = self._conv_2_bn(mask_rois, training=training)
            mask_rois = self._conv_3(mask_rois)
            if self.use_gn:
                mask_rois = self._conv_3_gn(mask_rois)
            if self.use_bn:
                mask_rois = self._conv_3_bn(mask_rois, training=training)
            mask_rois = self._deconv(mask_rois)
            mask_rois = self._masks(mask_rois)
            mask_rois = tf.transpose(mask_rois, [0, 3, 1, 2])
            mask_rois = tf.expand_dims(mask_rois, [-1])
            mask_list.append(mask_rois)
        return mask_list
        
    @tf.function(experimental_relax_shapes=True)
    def gather_mask_predictions(self, pred_mask, rcnn_target_matchs):
        pred_mask = tf.boolean_mask(pred_mask, rcnn_target_matchs!=0)
        mask_indices = tf.range(tf.shape(rcnn_target_matchs)[0])
        mask_indices = tf.transpose(tf.stack([mask_indices, rcnn_target_matchs-1]))
        mask_indices = tf.boolean_mask(mask_indices, rcnn_target_matchs!=0)
        pred_mask = tf.gather_nd(pred_mask, mask_indices)
        return pred_mask

    @tf.function(experimental_relax_shapes=True)
    def crop_masks(self, rois, fg_assignments, gt_masks, img_metas, size=(28, 28)):
        H = tf.reduce_mean(img_metas[...,6])
        W = tf.reduce_mean(img_metas[...,7])
        norm_rois = tf.concat(rois, axis=0) / tf.stack([H, W, H ,W])
        cropped_masks = tf.image.crop_and_resize(gt_masks,
                             norm_rois,
                             fg_assignments,
                             size,
                             method='nearest')
        return cropped_masks

    @tf.function(experimental_relax_shapes=True)
    def _mask_loss_single(self, masks_pred, rcnn_target_matchs, rois, 
                      fg_assignments, gt_masks, img_metas):
        masks_pred = self.gather_mask_predictions(masks_pred, rcnn_target_matchs)
        masks_true = self.crop_masks(rois, fg_assignments, gt_masks, img_metas)
        masks_true = tf.boolean_mask(masks_true, rcnn_target_matchs!=0)
        loss = tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=masks_true, 
                                                                          logits=masks_pred))
        mask_count = tf.shape(masks_pred)[0]
        return loss, mask_count

    @tf.function(experimental_relax_shapes=True)
    def mask_loss(self, masks_pred_list, rcnn_target_matchs, rois_list, 
                      fg_assignments, gt_masks, img_metas):
        batch_size = tf.shape(img_metas)[0]
        num_rois = tf.shape(rois_list[0])[0]
        fg_assignments = tf.cast(tf.reshape(fg_assignments, [batch_size, num_rois]), tf.int32)
        rcnn_target_matchs = tf.reshape(rcnn_target_matchs, [batch_size, num_rois])
        gt_masks = tf.expand_dims(gt_masks, [-1])
        loss = 0.
        mask_count = 0
        valid_losses = 0
        for i in range(img_metas.shape[0]):
            single_loss, count=self._mask_loss_single(masks_pred_list[i], rcnn_target_matchs[i],
                                    rois_list[i], fg_assignments[i], gt_masks[i],
                                    img_metas[i])
            # if no masks detected, don't add to loss
            if tf.math.is_nan(single_loss):
                continue
            valid_losses += 1
            loss += single_loss
            mask_count += count
        # adjust in case we got any nan value
        nan_multiplier = tf.cast(img_metas.shape[0]/valid_losses, loss.dtype)
        loss *= nan_multiplier
        mask_count = tf.cast(mask_count*tf.math.multiply(self.crop_size[0], 
                                                         self.crop_size[1]), loss.dtype)
        mask_count *= nan_multiplier
        loss /= mask_count
        loss *= tf.cast(batch_size, loss.dtype)
        return loss
    
    @tf.function(experimental_relax_shapes=True)
    def mold_masks(self, masks, bboxes, img_meta, threshold=0.5):
        mask_array = tf.TensorArray(tf.int32, size=tf.shape(masks)[0])
        bboxes = tf.cast(bboxes, tf.int32)
        img_meta = tf.cast(img_meta, tf.int32)
        for idx in tf.range(100):
            mask_array = mask_array.write(idx, self._mold_single_mask(masks[idx], bboxes[idx], img_meta, threshold))
        mask_array = mask_array.stack()
        return mask_array

    @tf.function(experimental_relax_shapes=True)
    def _mold_single_mask(self, mask, bbox, img_meta, threshold=0.5):
        '''
        Resize a mask and paste to background for image
        '''
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2] 
        x2 = bbox[3]
        h = y2 - y1
        w = x2 - x1
        if tf.math.multiply(h, w)<=0:
            return tf.zeros((img_meta[6], img_meta[7], 1), dtype=tf.int32)
        mask = tf.math.sigmoid(mask)
        mask_resize = tf.cast(tf.image.resize(mask, (h, w))>threshold, tf.int32)
        pad = [[y1, img_meta[6]-y2], [x1, img_meta[7]-x2], [0,0]]
        mask_resize = tf.pad(mask_resize, pad)
        return mask_resize
    
    # This is a different way to compute loss across all batches
    @tf.function(experimental_relax_shapes=True)
    def batch_indices(self, fg_assignments, rcnn_target_matchs, img_metas):
        batch_size = tf.shape(img_metas)[0]
        batch_length = tf.shape(fg_assignments)[0]//batch_size
        batch_indices = tf.gather(tf.cast(tf.repeat(tf.range(batch_size), batch_length), tf.int32) * 1, 
                                  tf.squeeze(tf.where(rcnn_target_matchs!=0)))
        gt_indices = tf.cast(tf.gather(fg_assignments * 1, 
                                       tf.squeeze(tf.where(rcnn_target_matchs!=0))), tf.int32)
        gt_indices = tf.transpose(tf.stack([batch_indices, gt_indices]))
        if tf.rank(gt_indices) == 1:
            gt_indices = tf.expand_dims(gt_indices, axis=0)
        return gt_indices
    
    @tf.function(experimental_relax_shapes=True)
    def fg_masks(self, gt_masks, fg_assignments, rcnn_target_matchs, img_metas):
        return tf.expand_dims(tf.gather_nd(gt_masks * 1, self.batch_indices(fg_assignments, rcnn_target_matchs, img_metas)), axis=-1)

    @tf.function(experimental_relax_shapes=True)
    def crop_and_resize(self, rois_list, gt_masks, fg_assignments, rcnn_target_matchs, img_metas):
        H = img_metas[0,6]
        W = img_metas[0,7]
        norm_rois = tf.gather(tf.concat(rois_list, axis=0) * 1, 
                                tf.squeeze(tf.where(rcnn_target_matchs!=0))) \
                                / tf.stack([H, W, H ,W])
        if tf.rank(norm_rois)==1:
            norm_rois = tf.expand_dims(norm_rois, axis=0)
        fg_masks = self.fg_masks(gt_masks, fg_assignments, rcnn_target_matchs, img_metas)
        cropped_targets = tf.image.crop_and_resize(fg_masks, 
                             norm_rois, 
                             tf.range(tf.shape(norm_rois)[0]), 
                             self.crop_size)
        #cropped_targets = tf.cast(cropped_targets, tf.int32)
        return cropped_targets
    
    @tf.function(experimental_relax_shapes=True)
    def mask_gather(self, rcnn_masks, fg_assignments, rcnn_target_matchs, img_metas):
        batch_size = tf.shape(img_metas)[0]
        batch_length = tf.shape(fg_assignments)[0]//batch_size
        batch_indices = tf.repeat(tf.range(batch_size), batch_length)
        batch_positions = tf.tile(tf.range(batch_length), [batch_size])
        mask_positions = tf.stack([batch_indices, batch_positions, rcnn_target_matchs - 1])
        mask_positions = tf.transpose(mask_positions)
        mask_positions = tf.boolean_mask(mask_positions, rcnn_target_matchs!=0)
        return tf.gather_nd(tf.stack(rcnn_masks), mask_positions)
    
    @tf.function(experimental_relax_shapes=True)
    def loss(self, rcnn_masks, fg_assignments, rcnn_target_matchs, 
             rois_list, gt_masks, img_metas):
        batch_size = tf.shape(img_metas)[0]
        target = self.crop_and_resize(rois_list, gt_masks, fg_assignments, 
                                      rcnn_target_matchs, img_metas)
        mask_pred = self.mask_gather(rcnn_masks, fg_assignments, 
                                rcnn_target_matchs, img_metas)
        loss = tf.reduce_mean(self.loss_func(target, mask_pred))
        batch_size = tf.cast(batch_size, loss.dtype)
        loss *= batch_size
        return loss
    
    