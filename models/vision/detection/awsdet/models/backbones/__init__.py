# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from .keras_backbone import KerasBackbone
from .resnet_common import ResNet50, ResNet50V2, ResNet101
from .resnet_aws import ResNet50V1

__all__ = ['KerasBackbone', 'ResNet50', 'ResNet50V1', 'ResNet50V2', 'ResNet101']

