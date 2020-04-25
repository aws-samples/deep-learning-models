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
             interval=1000,
             threshold=0.75,
             figsize=(8, 8),
             top_k=10):
        self.dataset = datasets.build_dataset(dataset_cfg)
        self.tf_dataset, self.num_examples = datasets.build_dataloader(self.dataset, 
                                                             1, 1, 
                                                             num_gpus=1, dist=False)
        self.tf_dataset = iter(self.tf_dataset.prefetch(16).shuffle(4).repeat())
        self.interval = interval
        self.img_mean = dataset_cfg.mean
        self.threshold = threshold
        self.figsize = figsize
        self.top_k = top_k
        self.threads = ThreadPoolExecutor()
    
    @tf.function
    def get_prediction(self, img, meta, model):
        result = model((img, meta), training=False)
        top_boxes = tf.gather_nd(result['bboxes'], 
                                 tf.where(result['scores']>=self.threshold))
        top_classes = tf.gather_nd(result['labels'],
                                   tf.where(result['scores']>=self.threshold))
        top_scores = tf.gather_nd(result['scores'], 
                                  tf.where(result['scores']>=self.threshold))
        # if no results, grab the top k predictions
        if tf.shape(top_boxes)[0]==0:
            top_boxes = result['bboxes'][:self.top_k]
            top_classes = result['labels'][:self.top_k]
            top_scores = result['scores'][:self.top_k]
        original_image = img[0][:int(meta[0][3]), :int(meta[0][4])]
        original_image = (tf.reverse(original_image, axis=[-1])+self.img_mean)
        return original_image, top_boxes, top_classes, top_scores
    
    @master_only
    def before_run(self, runner):
        img, meta = next(self.tf_dataset)
        original_image, top_boxes, top_classes, \
        top_scores = self.get_prediction(img, meta, runner.model)
    
    @master_only
    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            img, meta = next(self.tf_dataset)
            original_image, top_boxes, top_classes, \
            top_scores = self.get_prediction(img, meta, runner.model)
            image_thread = self.threads.submit(self.make_image_thread, runner, 
                                        original_image, top_boxes, 
                                        top_classes, top_scores)
            
    def make_image_thread(self, runner, original_image, top_boxes, 
                          top_classes, top_scores):
        image = visualize.make_image(original_image.numpy(), 
                                         top_boxes.numpy(), 
                                         top_classes.numpy(),
                                         visualize.coco_categories, 
                                         figsize=self.figsize, 
                                         scores=top_scores.numpy())
        image = np.expand_dims(image/255, axis=0)
        runner.log_buffer.update({'prediction_image': image})
