# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import numpy as np



class DataGenerator:

    def __init__(self, dataset, num_gpus=0, index=0, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.gpu_index = index
        self.num_gpus = num_gpus
        if num_gpus > 0:
            self.num_examples = len(dataset) // num_gpus
        else:
            self.num_examples = len(dataset)
 
    def __call__(self):
        if self.num_gpus == 0:
            indices = np.arange(0, len(self.dataset))
        else:
            if self.dataset.train:
                indices = np.arange(0, len(self.dataset)) # ensure that each worker has a different seed
            else:
                indices = np.arange(self.gpu_index, len(self.dataset), self.num_gpus)
        while True:
            if self.shuffle:
                np.random.shuffle(indices)

            print('Starting new loop for GPU:', self.gpu_index)
            for img_idx in indices:
                # overfit hack  DEBUG
                # img_idx = self.gpu_index * 100 + img_idx % 100 #DEBUG
                if self.dataset.train:
                    if self.dataset.mask:
                        img, img_meta, bboxes, labels, mask = self.dataset[img_idx]
                        yield img, img_meta, bboxes, labels, mask
                    else:
                        img, img_meta, bboxes, labels = self.dataset[img_idx]
                        # print(self.gpu_index, img_meta)
                        yield img, img_meta, bboxes, labels
                else:
                    img, img_meta = self.dataset[img_idx]
                    yield img, img_meta
