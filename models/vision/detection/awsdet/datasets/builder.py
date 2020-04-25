# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import copy

from awsdet.utils import build_from_cfg
from .registry import DATASETS

def build_dataset(cfg, default_args=None):
    #TODO: Handle cases with multiple datasets, etc.
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset
