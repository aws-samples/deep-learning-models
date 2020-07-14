# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import numpy as np

from awsdet.datasets.utils import (imrescale, imnormalize, img_flip, 
                                  impad_to_multiple, impad_to_square,
                                  impad_mask_to_square,
                                  impad_mask_to_multiple,
                                  bbox_flip)

class ImageTransform(object):
    '''Preprocess the image.
    
        1. rescale the image to expected size
        2. normalize the image
        3. flip the image (if needed)
        4. pad the image (if needed)
    '''
    def __init__(self,
                 scale=(800, 1333),
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 pad_mode='fixed'):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.pad_mode = pad_mode
        self.impad_size = max(scale) if pad_mode == 'fixed' else 64

    def __call__(self, img, flip=False):
        img, scale_factor = imrescale(img, self.scale)
        img_shape = img.shape
        img = imnormalize(img, self.mean, self.std)
          
        if flip:
            img = img_flip(img)
        if self.pad_mode == 'fixed':
            img = impad_to_square(img, self.impad_size)

        else: # 'non-fixed'
            img = impad_to_multiple(img, self.impad_size)
        
        return img, img_shape, scale_factor

class BboxTransform(object):
    '''Preprocess ground truth bboxes.
    
        1. rescale bboxes according to image size
        2. flip bboxes (if needed)
    '''
    def __init__(self):
        pass
    
    def __call__(self, bboxes, labels, 
                 img_shape, scale_factor, flip=False):
 
        bboxes = bboxes * scale_factor
        if flip:
            bboxes = bbox_flip(bboxes, img_shape)
            
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[0])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[1])
            
        return bboxes, labels

class MaskTransform(object):
    '''
    Preprocess a mask to match
    the preprocessed image
    1. rescale to match image
    2. flip if needed
    3. pad image
    '''
    def __init__(self,
                 scale=(800, 1333),
                 pad_mode='fixed'):
        self.scale = scale
        self.pad_mode = pad_mode
        self.impad_size = max(scale) if pad_mode == 'fixed' else 64
    
    def __call__(self, mask, flip=False):
        mask, scale_factor = imrescale(mask, self.scale)
        if flip:
            mask = img_flip(mask)
        if self.pad_mode == 'fixed':
            mask = impad_mask_to_square(mask, self.impad_size)
        else:
            mask = impad_mask_to_multiple(mask, self.impad_size)
        return mask