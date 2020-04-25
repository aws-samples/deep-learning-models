# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import get_classes
from mmdet.models import build_detector
from mmdet.utils.misc import Config
from mmdet.utils import generic, image, visualization
from mmdet.utils.image import io


import os
import os.path as osp

import numpy as np

from pycocotools.cocoeval import COCOeval
from mmdet.utils.runner import Hook
from mmdet import datasets
from mmdet.core.evaluation.coco_utils import fast_eval_recall, results2json
from mmdet.core.evaluation.mean_ap import eval_map
from mmdet.utils.misc import ProgressBar
from mmdet.datasets import build_dataloader
from mmdet.utils.fileio import load, dump
from mmdet.core.bbox import transforms
import tensorflow as tf


def load_checkpoint(model, filename):
    return model.load_weights(filepath=filename)

def init_detector(config, checkpoint=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`Config`): Config file path or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        tf.keras.Model instance
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, train_cfg=None, test_cfg=config.test_cfg)
    # dummy data to init network
    img = tf.random.uniform(shape=[1216, 1216, 3], dtype=tf.float32)
    img_meta = tf.constant(
        [465., 640., 3., 800., 1101., 3., 1216., 1216., 3., 1.7204301, 0.],
        dtype=tf.float32)
    _ = model((tf.expand_dims(img, axis=0), tf.expand_dims(img_meta, axis=0)),
              training=False)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
    return model


def evaluate(results, dataset):
    tmp_file = osp.join('offline_temp_0')
    result_files = results2json(dataset, results, tmp_file)
    #res_types = ['bbox', 'segm'
    #            ] if runner.model.module.with_mask else ['bbox']
    res_types = ['bbox']
    cocoGt = dataset.coco
    imgIds = cocoGt.getImgIds()
    for res_type in res_types:
        try:
            cocoDt = cocoGt.loadRes(result_files[res_type])
        except IndexError:
            print('No prediction found.')
            break
        iou_type = res_type
        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    #     metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    #     for i in range(len(metrics)):
    #         key = '{}_{}'.format(res_type, metrics[i])
    #         val = float('{:.3f}'.format(cocoEval.stats[i]))
    #         runner.log_buffer.output[key] = val
    #     runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
    #         '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
    #         '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
    # runner.log_buffer.ready = True
    # for res_type in res_types:
    #     os.remove(result_files[res_type])


def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (tf.keras.Model): A Keras model.
        data: Tuple of padded batch data - batch_imgs, batch_metas, batch_bboxes, batch_labels
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    detections = model(data, training=train_mode)  #TODO write a better interface
    outputs = dict(num_samples=data[0].shape[0])
    outputs.update(detections)
    return outputs


def offline_evaluation(dataset, model):
    '''
    Single gpu evaluation of checkpoints on test data
    '''
    # create a loader for this runner
    tf_dataset, num_examples = build_dataloader(dataset, 1, 1, num_gpus=0, dist=False)
    results = [None for _ in range(num_examples)]

    for i, data_batch in enumerate(tf_dataset):
        print('Processing image id', dataset.img_ids[i])
        if i >= num_examples:
            break
        _, img_meta = data_batch
        outputs = batch_processor(model, data_batch, train_mode=False)
        assert isinstance(outputs, dict)
        bboxes = outputs['bboxes']
        # map boxes back to original scale
        bboxes = transforms.bbox_mapping_back(bboxes, img_meta)
        labels = outputs['labels']
        cat_ids = [dataset.cat_ids[l] for l in labels]
        scores = outputs['scores']
        print(bboxes, cat_ids, scores)
        result = transforms.bbox2result(bboxes, labels, scores, num_classes=dataset.CLASSES+1)
        results[i] = result

    # run coco eval
    evaluate(results, dataset)


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (tf.keras.Model): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    raise NotImplementedError

def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = io.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    if segm_result is not None:
        segms = generic.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    visualization.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img


def show_result_pyplot(img,
                       result,
                       class_names,
                       score_thr=0.3,
                       fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    img = show_result(
        img, result, class_names, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(image.transforms.bgr2rgb(img))


if __name__ == "__main__":
    base_dir = '/Users/mzanur/workspace/mmdetection_tf/'
    config_file = base_dir + 'configs/inference_cfg.py'
    model = init_detector(config_file, checkpoint=base_dir+'weights/intermediate/faster_rcnn') # faster_rcnn_resnet101.tf')
    dataset_cfg = dict(
        type='CocoDataset',
        train=False,
        dataset_dir='/Users/mzanur/data/COCO/',
        subset='train',
        flip_ratio=0,
        pad_mode='fixed',
        mean=(123.675, 116.28, 103.53),
        std=(1., 1., 1.),
        scale=(800, 1216))
    dataset = datasets.build_dataset(dataset_cfg)
    offline_evaluation(dataset, model)
