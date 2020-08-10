# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import os
import os.path as osp
import time
import pathlib
import tarfile
import numpy as np
import tensorflow as tf

from awsdet.utils.misc import Config, mkdir_or_exist
from awsdet.utils.runner import init_dist, master_only, get_dist_info, get_barrier
from awsdet.utils.keras import freeze_model_layers

from awsdet import __version__
from awsdet.apis import (get_root_logger, set_random_seed, train_detector,)
from awsdet.datasets import build_dataset, build_dataloader
from awsdet.models import build_detector

#tf.config.experimental_run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices('GPU')

##### TENSORFLOW RUNTIME OPTIONS #####

# tf.config.experimental_run_functions_eagerly(True)
os.environ['TF_CUDNN_USE_AUTOTUNE']= str(0)
os.environ['TF_DETERMINISTIC_OPS'] = str(1)
os.environ['PYTHONHASHSEED']=str(17)
os.environ['HOROVOD_FUSION_THRESHOLD']=str(0)

# init distributed env first
init_dist()

# avoid large pool of Eigen threads
tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(max(2, 40 // get_dist_info()[2]))
# reduce TF warning verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument("--model_dir", help="Location of model on Sagemaker instance")
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='restarts training from saved running state in provided directory')
    parser.add_argument('--resume_dir', help='restarts training from the latest running state in provided directory - useful for spot training')
    parser.add_argument('--amp', type=str2bool, nargs='?', const=True, default=True, help='enable mixed precision training')
    parser.add_argument('--validate', type=str2bool, nargs='?', const=True, default=True, help='whether to evaluate the checkpoint during training')
    parser.add_argument('--seed', type=int, default=17, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=True, help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--autoscale-lr', type=str2bool, nargs='?', const=True, default=True, help='automatically scale lr with the number of gpus')
    args = parser.parse_args()

    return args

 
def decompress_data():
    if get_dist_info()[1]==0:
        print("Decompressing Data")
        coco_tar = tarfile.open(pathlib.Path(os.getenv('SM_CHANNEL_COCO')).joinpath('coco.tar').as_posix())
        coco_tar.extractall(path=os.getenv('SM_CHANNEL_COCO'))
    # block other ranks form skipping ahead before data is ready
    barrier = get_barrier()


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


def main_ec2(args, cfg):
    # start logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    """
    Main training entry point for jobs launched directly on EC2 instances
    """
    num_gpus = len(gpus)
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.resume_dir is not None:
        if os.path.exists(args.resume_dir):
            logger.info("RESUMING TRAINING")
            # get the latest checkpoint
            all_chkpt = [os.path.join(args.resume_dir,d) for d in os.listdir(args.resume_dir) if os.path.isdir(os.path.join(args.resume_dir,d))]
            if not all_chkpt:
               cfg.resume_from = None
            else: 
               latest_chkpt = max(all_chkpt, key=os.path.getmtime)
               # set the latest checkpoint to resume_from
               cfg.resume_from = latest_chkpt
        else:
            logger.info("CHECKPOINT NOT FOUND, RESTARTING TRAINING")
            cfg.resume_from = None

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        total_bs = get_dist_info()[2] * cfg.data.imgs_per_gpu
        cfg.optimizer['learning_rate'] = cfg.optimizer['learning_rate'] * total_bs / 8

     # init distributed env first, since logger depends on the dist info.
     # init_dist()

    if not gpus:
        distributed = False  # single node single gpu
    else:
        distributed = True

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
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
    #model.save('my_model')
    # print('BEFORE:', model.layers[0].layers[0].get_weights()[0][0,0,0,:])
    weights_path = cfg.model['backbone']['weights_path']
    logger.info('Loading weights from: {}'.format(weights_path))
    if osp.splitext(weights_path)[1] == '.h5': # older keras format from Keras model zoo
        model.layers[0].layers[0].load_weights(weights_path, by_name=True, skip_mismatch=True)
    else: # SavedModel format assumed - extract weights
        backbone_model = tf.keras.models.load_model(weights_path)
        # load weights if layers match
        for layer_idx, layer in enumerate(backbone_model.layers):
            if layer_idx < len(model.layers[0].layers[0].layers):
                model.layers[0].layers[0].layers[layer_idx].set_weights(layer.get_weights())
                print('Loaded weights for:', layer.name)
        del backbone_model
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

    
def main_sagemaker(args, cfg):
    """
    Main training entry point for jobs launched via SageMaker
    """
    instance_name = cfg.sagemaker_job['job_name']
    s3_path = cfg.sagemaker_job['s3_path']
    
    decompress_data() # setup data dirs based on SM CHANNELS

    num_gpus = len(gpus)
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        total_bs = get_dist_info()[2] * cfg.data.imgs_per_gpu
        cfg.optimizer['learning_rate'] = cfg.optimizer['learning_rate'] * total_bs / 8

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

    # sagemaker specific path resolution
    import os, pathlib
    data_root = pathlib.Path(os.getenv('SM_CHANNEL_COCO')).joinpath('coco').as_posix()
    cfg.data.train['dataset_dir'] = data_root
    cfg.data.val['dataset_dir'] = data_root
    weights_file = cfg.model['backbone']['weights_path']
    weights_path = pathlib.Path(os.getenv('SM_CHANNEL_WEIGHTS')).joinpath(weights_file).as_posix()
    logger.info('Loading weights from: {}'.format(weights_path))
    if osp.splitext(weights_file)[1] == '.h5': # older keras format from Keras model zoo
        model.layers[0].layers[0].load_weights(weights_path, by_name=True, skip_mismatch=True)
    else: # SavedModel format assumed - extract weights
        backbone_model = tf.keras.models.load_model(weights_path)
        # load weights if layers match
        for layer_idx, layer in enumerate(backbone_model.layers):
            if layer_idx < len(model.layers[0].layers[0].layers):
                model.layers[0].layers[0].layers[layer_idx].set_weights(layer.get_weights())
                print('Loaded weights for:', layer.name)
        del backbone_model
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
    args = parse_args()
    cfg = Config.fromfile(args.config)
    train_on_sagemaker = cfg.train_cfg.get('sagemaker', False)
    if train_on_sagemaker:
        main_sagemaker(args, cfg)
    else:
        main_ec2(args, cfg)
