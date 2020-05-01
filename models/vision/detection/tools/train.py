# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import os
import os.path as osp
import time

import numpy as np
import tensorflow as tf

from awsdet.utils.misc import Config, mkdir_or_exist
from awsdet.utils.runner import init_dist, master_only, get_dist_info
from awsdet.utils.keras import freeze_model_layers

from awsdet import __version__
from awsdet.apis import (get_root_logger, set_random_seed, train_detector,)
from awsdet.datasets import build_dataset, build_dataloader
from awsdet.models import build_detector

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental_run_functions_eagerly(True)
os.environ['TF_CUDNN_USE_AUTOTUNE']= str(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        'config',
        help='train config file path'
    )
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from',
                        help='the checkpoint file to resume from')
    parser.add_argument('--amp', action='store_true', help='enable mixed precision training')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--seed', type=int, default=17, help='random seed'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--autoscale-lr',
                        action='store_true',
                        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()

    return args


@master_only
def print_model_info(model, logger):

    # logger.info('Model structure:')
    def print_nested(l, level=1):
        logger.info('{}{}\ttrainable: {}'.format('\t' * level, l.name,
                                                 l.trainable))
        # print(l.get_weights())
        if hasattr(l, 'layers'):
            for ll in l.layers:
                print_nested(ll, level + 1)

    for l in model.layers:
        print_nested(l)

    logger.info('Trainable Variables:')
    for var in model.trainable_variables:
        logger.info(var.name)
    # model summary does not work for subclassed models
    # model.summary(line_length=80)

def main():
    args = parse_args()
    num_gpus = len(gpus)
    cfg = Config.fromfile(args.config)
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        total_bs = len(gpus) * cfg.data.imgs_per_gpu
        cfg.optimizer['learning_rate'] = \
            cfg.optimizer['learning_rate'] * total_bs / 8

    # init distributed env first, since logger depends on the dist info.
    init_dist()

    if not gpus:
        distributed = False  # single node single gpu
    else:
        distributed = True

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('TF MMDetection Version: {}'.format(__version__))
    logger.info('Config:\n{}'.format(cfg.text))
    logger.info('Tensorflow version: {}'.format(tf.version.VERSION))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed + get_dist_info()[0], deterministic=args.deterministic)

    model = build_detector(cfg.model,
                           train_cfg=cfg.train_cfg,
                           test_cfg=cfg.test_cfg)

    # dummy data to init network
    padded_img_side = max(cfg.data.train['scale'])
    img = tf.random.uniform(shape=[padded_img_side, padded_img_side, 3], dtype=tf.float32)
    img_meta = tf.constant(
        [465., 640., 3., 800., 1101., 3., float(padded_img_side), float(padded_img_side), 3., 1.7204301, 0.],
        dtype=tf.float32)
    # bboxes = tf.constant([[1.0, 1.0, 10.0, 10.0]], dtype=tf.float32)
    # labels = tf.constant([1], dtype=tf.int32)
    _ = model((tf.expand_dims(img, axis=0), tf.expand_dims(img_meta, axis=0)),
              training=False)

    # print('BEFORE:', model.layers[0].layers[0].get_weights()[0][0,0,0,:])
    weights_path = cfg.model['backbone']['weights_path']
    logger.info('Loading weights from: {}'.format(weights_path))
    model.layers[0].layers[0].load_weights(weights_path, by_name=True, skip_mismatch=True) #by_name=False)
    # print('AFTER:',model.layers[0].layers[0].get_weights()[0][0,0,0,:])

    print_model_info(model, logger)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) > 1:
        raise NotImplementedError

    train_detector(model,
                   datasets,
                   cfg,
                   num_gpus=num_gpus,
                   distributed=distributed,
                   mixed_precision=args.amp,
                   validate=args.validate,
                   timestamp=timestamp)


if __name__ == '__main__':
    main()

