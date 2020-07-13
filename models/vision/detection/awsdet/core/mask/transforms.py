import tensorflow as tf
import numpy as np
import cv2
from collections import defaultdict
import pycocotools.mask as mask_util

@tf.function(experimental_relax_shapes=True)
def mold_masks(masks, bboxes, img_meta, threshold=0.5):
    mask_array = tf.TensorArray(tf.int32, size=tf.shape(masks)[0])
    bboxes = tf.cast(bboxes, tf.int32)
    img_meta = tf.cast(img_meta, tf.int32)
    for idx in tf.range(100):
        mask_array = mask_array.write(idx, self._mold_single_mask(masks[idx], bboxes[idx], img_meta, threshold))
    mask_array = mask_array.stack()
    return mask_array

@tf.function(experimental_relax_shapes=True)
def _mold_single_mask(mask, bbox, img_meta, threshold=0.5):
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
    mask_resize = tf.cast(tf.image.resize(mask, (h, w), method='nearest')>threshold, tf.int32)
    pad = [[y1, img_meta[6]-y2], [x1, img_meta[7]-x2], [0,0]]
    mask_resize = tf.pad(mask_resize, pad)
    return mask_resize

def mask2result(masks, labels, meta, num_classes=81, threshold=0.5):
    meta = np.squeeze(meta)
    img_heights, img_widths = meta[:2].astype(np.int32)
    unpadded_height = tf.cast(meta[3], tf.int32)
    unpadded_width = tf.cast(meta[4], tf.int32)
    orig_height = tf.cast(meta[0], tf.int32)
    orig_width = tf.cast(meta[1], tf.int32)
    masks = masks[:,:unpadded_height,:unpadded_width, :]
    masks = tf.image.resize(masks, (orig_height, orig_width), method='nearest')
    masks_np = np.squeeze((masks.numpy()>threshold).astype(np.int32))
    labels_np = labels.numpy()
    if meta[-1]==1:
        masks_np = np.flip(masks_np, axis=2)
    lists = defaultdict(list)
    for i,j in enumerate(labels_np):
        lists[j].append(mask_util.encode(
                    np.array(
                        masks_np[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])
    return lists