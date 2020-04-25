# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from .builder import build_dataset
from .coco import CocoDataset
from .registry import DATASETS
from .loader.build_loader import build_dataloader

__all__ = ['DATASETS', 'build_dataset', 'build_dataloader', 'CocoDataset']
