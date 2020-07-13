# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from .hook import Hook
from awsdet.utils import visualize
from awsdet import datasets
from awsdet.utils.runner.dist_utils import master_only
from concurrent.futures import ThreadPoolExecutor

class Visualizer(Hook):
    
    def __init__(self,
             dataset_cfg,
             interval=200,
             threshold=0.75,
             figsize=(8, 8),
             top_k=10,
             run_on_sagemaker=False):
        if run_on_sagemaker:
            # update paths for SM
            import os, pathlib
            data_root = pathlib.Path(os.getenv('SM_CHANNEL_COCO')).joinpath('coco').as_posix()
            dataset_cfg['dataset_dir'] = data_root
        self.dataset_cfg = dataset_cfg
        self.img_mean = dataset_cfg['mean']
        self.img_std = dataset_cfg['std']
        self.dataset = datasets.build_dataset(dataset_cfg)
        self.tf_dataset, self.num_examples = datasets.build_dataloader(self.dataset, 1, 1, num_gpus=1, dist=False)
        self.tf_dataset = iter(self.tf_dataset.prefetch(16).shuffle(4).repeat())
        self.interval = interval
        self.img_mean = dataset_cfg.mean
        self.threshold = threshold
        self.figsize = figsize
        self.top_k = top_k
        self.threads = ThreadPoolExecutor()
    
    #@tf.function
    def get_prediction(self, img, meta, model):
        result = model((img, meta), training=False)
        original_image = img[0][:int(meta[0][3]), :int(meta[0][4])]
        #original_image = (tf.reverse(original_image, axis=[-1])+self.img_mean)
        original_image *= self.img_std
        original_image += self.img_mean
        if 'masks' in result.keys():
            result['masks'] = result['masks'][:, :int(meta[0][3]), :int(meta[0][4]), :]
        detection_dict = {}
        detection_dict['top_boxes'] = tf.gather_nd(result['bboxes'], 
                                 tf.where(result['scores']>=self.threshold))
        detection_dict['top_classes'] = tf.gather_nd(result['labels'],
                                       tf.where(result['scores']>=self.threshold))
        detection_dict['top_scores'] = tf.gather_nd(result['scores'], 
                                      tf.where(result['scores']>=self.threshold))
        if 'masks' in result.keys():
            detection_dict['top_masks'] = tf.gather_nd(result['masks'],
                                     tf.where(result['scores']>=self.threshold))
        if tf.shape(detection_dict['top_boxes'])[0]==0:
            detection_dict['top_boxes'] = result['bboxes'][:self.top_k]
            detection_dict['top_classes'] = result['labels'][:self.top_k]
            detection_dict['top_scores'] = result['scores'][:self.top_k]
            if 'masks' in result.keys():
                 detection_dict['top_masks'] = result['masks'][:self.top_k]
        return original_image, detection_dict
    
    @master_only
    def before_run(self, runner):
        img, meta = next(self.tf_dataset)
        original_image, \
        detection_dict = self.get_prediction(img, meta, runner.model)
    
    @master_only
    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            img, meta = next(self.tf_dataset)
            original_image, \
            detection_dict = self.get_prediction(img, meta, runner.model)
            image_thread = self.threads.submit(self.make_image_thread, runner, 
                                        original_image, detection_dict)
            
    def make_image_thread(self, runner, original_image, detection_dict):
        if 'top_masks' in detection_dict.keys():
            masks = detection_dict['top_masks'].numpy()
        else:
            masks = None
        boxes = detection_dict['top_boxes'].numpy()
        classes = detection_dict['top_classes'].numpy()
        scores = detection_dict['top_scores'].numpy()
        image = visualize.make_image(original_image.numpy(), 
                                         boxes, 
                                         classes,
                                         visualize.coco_categories, 
                                         figsize=self.figsize, 
                                         scores=scores,
                                         masks=masks)
        image = np.expand_dims(image/255., axis=0)
        if 'top_masks' in detection_dict.keys():
            combined_mask = np.expand_dims(np.max(detection_dict['top_masks'].numpy(), axis=0), axis=0)
            combined_mask = combined_mask.astype(np.float32)
            runner.log_buffer.update({'image_mask': combined_mask})
        runner.log_buffer.update({'prediction_image': image})
        
