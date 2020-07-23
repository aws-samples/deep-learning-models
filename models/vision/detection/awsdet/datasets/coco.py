# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import os.path as osp
import cv2
import numpy as np
from pycocotools.coco import COCO
from .registry import DATASETS
from . import transforms, utils


@DATASETS.register_module
class CocoDataset(object):
    def __init__(self,
                 dataset_dir,
                 subset,
                 flip_ratio=0,
                 pad_mode='fixed',
                 mean=(0., 0., 0.),
                 std=(1., 1., 1.),
                 preproc_mode='caffe',
                 scale=(1024, 800),
                 train=False,
                 debug=False,
                 mask=False):
        """
        Load a subset of the COCO dataset.
        
        Args:
            dataset_dir: The root directory of the COCO dataset.
            subset: What to load (train, val).
            flip_ratio: Float. The ratio of flipping an image and its bounding boxes.
            pad_mode: Which padded method to use (fixed, non-fixed)
            mean: Tuple. Image mean.
            std: Tuple. Image standard deviation.
            scale: Tuple of two integers.
        Returns:
            A COCODataset instance
        """

        if subset not in ['train', 'val']:
            raise AssertionError('subset must be "train" or "val".')

        self.coco = COCO("{}/annotations/instances_{}2017.json".format(
            dataset_dir, subset))

        # get the mapping from original category ids to labels
        self.cat_ids = self.coco.getCatIds()
        self.CLASSES = len(self.cat_ids)
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        self.img_ids, self.img_infos = self._filter_imgs()

        if debug:
            self.img_ids, self.img_infos = \
                self.img_ids[:50], self.img_infos[:50]

        self.image_dir = "{}/{}2017".format(dataset_dir, subset)

        self.flip_ratio = flip_ratio

        if pad_mode in ['fixed', 'non-fixed']:
            self.pad_mode = pad_mode
        elif subset == 'train':
            self.pad_mode = 'fixed'
        else:
            self.pad_mode = 'non-fixed'
        
        self.rgb_mean = mean
        self.rgb_std = std
        self.img_transform = transforms.ImageTransform(scale, mean, std,
                                                       pad_mode)
        self.bbox_transform = transforms.BboxTransform()
        self.mask_transform = transforms.MaskTransform(scale, pad_mode)
        self.train = train
        self.preproc_mode = preproc_mode
        self.mask = mask

    def _filter_imgs(self, min_size=32):
        """
        Filter images too small or without ground truths.
        
        Args:
            min_size: the minimal size of the image.
        """
        # Filter images without ground truths.
        all_img_ids = list(
            set([_['image_id'] for _ in self.coco.anns.values()]))
        # Filter images too small.
        img_ids = []
        img_infos = []
        for i in all_img_ids:
            info = self.coco.loadImgs(i)[0]

            ann_ids = self.coco.getAnnIds(imgIds=i)
            ann_info = self.coco.loadAnns(ann_ids)
            ann = self._parse_ann_info(ann_info)

            if min(info['width'],
                   info['height']) >= min_size and ann['labels'].shape[0] != 0:
                img_ids.append(i)
                img_infos.append(info)
        return img_ids, img_infos

    def _load_ann_info(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info

    def _parse_ann_info(self, ann_info):
        """
        Parse bbox annotation.
        
        Args:
            ann_info (list[dict]): Annotation info of an image.
            
        Returns:
            dict: A dict containing the following keys: bboxes, 
                bboxes_ignore, labels.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        for _, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [y1, x1, y1 + h, x1 + w]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(bboxes=gt_bboxes,
                   labels=gt_labels,
                   bboxes_ignore=gt_bboxes_ignore)

        return ann

    def __len__(self):
        return len(self.img_infos)


    def _tf_preprocessing(self, image):
        """
        [-1, 1] used by V2 implementations
        Args:
            image: numpy array
        Returns:
            Scaled image [-1.0, 1.0]
        """
        return image/127.0 - 1.0
 

    def _caffe_preprocessing(self, image):
        """
        BGR zero centered
        Args:
            image: numpy array
        Returns:
            Zero centered BGR image
        """
        pixel_means = self.rgb_mean[::-1]
        channels = cv2.split(image)
        for i in range(3):
            channels[i] -= pixel_means[i]
        return cv2.merge(channels)


    def _rgb_preprocessing(self, image):
        """
        RGB standardized
        Args:
            image: numpy array
        Returns:
            Standardized RGB image
        """
        channels = cv2.split(image)
        for i in range(3):
            channels[i] -= self.rgb_mean[i]
            channels[i] /= self.rgb_std[i]
        return cv2.merge(channels)


    def __getitem__(self, idx):
        """
        Load the image and its bboxes for the given index.
        
        Args:
            idx: the index of images.
        Returns:
            tuple: A tuple containing the following items: image, 
                bboxes, labels.
        """
        img_info = self.img_infos[idx]
        ann_info = self._load_ann_info(idx)

        # load the image.
        bgr_img = cv2.imread(osp.join(self.image_dir, img_info['file_name']), cv2.IMREAD_COLOR).astype(np.float32)
        if self.preproc_mode == 'tf':
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            img = self._tf_preprocessing(rgb_img)
        elif self.preproc_mode == 'caffe':
            img = self._caffe_preprocessing(bgr_img)
        elif self.preproc_mode == 'rgb':
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            img = self._rgb_preprocessing(rgb_img)
        else:
            raise NotImplementedError("Preprocessing mode '{}' not supported".format(self.preproc_mode))

        ori_shape = img.shape

        # Load the annotation.
        ann = self._parse_ann_info(ann_info)
        bboxes = ann['bboxes']
        labels = ann['labels']

        flip = True if np.random.rand() < self.flip_ratio else False
        
        if self.mask:
            masks = np.array([self.mask_transform(self.coco.annToMask(i), flip=flip) \
                     for i in ann_info])
            masks = masks.astype(np.int32)
        
        # Handle the image
        img, img_shape, scale_factor = self.img_transform(img, flip)

        pad_shape = img.shape

        # Handle the annotation.
        bboxes, labels = self.bbox_transform(bboxes, labels, img_shape, scale_factor, flip)

        # Handle the meta info.
        img_meta_dict = dict({
            'ori_shape': ori_shape,
            'img_shape': img_shape,
            'pad_shape': pad_shape,
            'scale_factor': scale_factor,
            'flip': flip
        })

        img_meta = utils.compose_image_meta(img_meta_dict)
        if self.train:
            if self.mask:
                return img, img_meta, bboxes, labels, masks
            return img, img_meta, bboxes, labels
        return img, img_meta


    def get_categories(self):
        """
        Get list of category names. 
        Returns:
            list: A list of category names.
        """
        # Note that the first item 'bg' means background.
        return ['bg'] + [self.coco.loadCats(i)[0]["name"] for i in self.cat2label.keys()]

