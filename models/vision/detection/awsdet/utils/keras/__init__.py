# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from .backbone import get_base_model, get_outputs
from .freeze import *
__all__ = ['get_base_model', 'get_outputs', 'freeze_model_layers']
