# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import logging
import random
import re
from collections import OrderedDict
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
from ..utils.runner import (Runner, get_dist_info, obj_from_dict)
from awsdet.core import CocoDistEvalmAPHook, CocoDistEvalRecallHook
#                        DistEvalmAPHook, DistOptimizerHook, Fp16OptimizerHook)
from awsdet.datasets import DATASETS, build_dataloader
#from awsdet.models import RPN


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('awsdet')
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=log_level)
    rank, _, _, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        # deterministic (bool): unused - to be enabled through TF FLAGS in 2.2.0 (see tools/train.py)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_losses(losses, local_batch_size):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if tf.is_tensor(loss_value):
            log_vars[loss_name] = tf.reduce_mean(loss_value)
        elif isinstance(loss_value, list):
            log_vars[loss_name] = tf.add_n(
                [tf.reduce_mean(_loss) for _loss in loss_value])
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))
    loss_list = []
    for _key, _value in log_vars.items():
        if 'loss' in _key:
            if 'reg_loss' not in _key:
                # https://github.com/horovod/horovod/issues/843
                # horovod averages (not sums) gradients by default over workers
                loss_list.append(_value/local_batch_size)
            else:
                loss_list.append(_value)
    total_loss = sum(loss_list) 
    log_vars['loss'] = total_loss
    return total_loss, log_vars


@tf.function(experimental_relax_shapes=True)
def batch_processor(model, data, train_mode, loss_weights=None):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (tf.keras.Model): A Keras model.
        data: Tuple of padded batch data - batch_imgs, batch_metas, batch_bboxes, batch_labels
        train_mode (bool): Training mode or not. It may be useless for some
            models.
        loss_weights: dictionary of weights that can be assigned in multiloss scenario, for example, {'rpn_class_loss': 1., 'rpn_bbox_loss': 1.,...} 

    Returns:
        dict: A dict containing losses and log vars.
    """
    if train_mode:
        losses = model(data, training=train_mode)
        # add regularization losses
        reg_losses = tf.add_n(model.losses)
        local_batch_size = data[0].shape[0]
        losses['reg_loss'] = reg_losses
        if not loss_weights is None:
            losses = {i:losses[i]*j for i,j in loss_weights.items()}
        loss, log_vars = parse_losses(losses, local_batch_size)
        outputs = dict(loss=loss,
                       log_vars=log_vars,
                       num_samples=local_batch_size)
    else:
        detections = model(data, training=train_mode)
        outputs = dict(num_samples=data[0].shape[0])
        outputs.update(detections)
    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   num_gpus=1,
                   distributed=False,
                   mixed_precision=False,
                   validate=False,
                   timestamp=None):
    logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model,
                    dataset,
                    cfg,
                    num_gpus=num_gpus,
                    mixed_precision=mixed_precision,
                    validate=validate,
                    logger=logger,
                    timestamp=timestamp)
    else:
        _non_dist_train(model,
                        dataset,
                        cfg,
                        validate=validate,
                        mixed_precision=mixed_precision,
                        logger=logger,
                        timestamp=timestamp)


def build_optimizer(optimizer_cfg):
    """Build optimizer from configs.

    Args:
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - learning_rate: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  momentum, etc.
                - paramwise_options: TODO

    Returns:
        tf.keras.Optimizer: The initialized optimizer.

    """
    optimizer_cfg = optimizer_cfg.copy()
    return obj_from_dict(optimizer_cfg, tf.keras.optimizers)


def _dist_train(model,
                dataset,
                cfg,
                num_gpus=1,
                mixed_precision=False,
                validate=False,
                logger=None,
                timestamp=None):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    tf_datasets = [
        build_dataloader(ds,
                         cfg.data.imgs_per_gpu,
                         1,
                         num_gpus=num_gpus,
                         dist=True) for ds in dataset
    ]

    # build runner
    optimizer = build_optimizer(cfg.optimizer)
    if mixed_precision:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scale='dynamic')

    optimizer_config = cfg.optimizer_config
    optimizer_config['amp_enabled'] = mixed_precision
    gradient_clip = optimizer_config.get('gradient_clip', 15.0) # default is 15.0

    runner = Runner(model,
                    batch_processor,
                    optimizer,
                    cfg.work_dir,
                    logger=logger,
                    amp_enabled=mixed_precision,
                    gradient_clip=gradient_clip)
 
    runner.timestamp = timestamp
    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    # register eval hooks
    if validate and runner.rank < runner.local_size: # register this dist eval hook only for Node 0
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)

    runner.run(tf_datasets, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model,
                    dataset,
                    cfg,
                    mixed_precision=False,
                    validate=False,
                    logger=None,
                    timestamp=None):
    if validate:
        raise NotImplementedError('Built-in validation is not implemented '
                                  'yet in not-distributed training. Use '
                                  'distributed training or test.py and '
                                  '*eval.py scripts instead.')

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    tf_datasets = [
        build_dataloader(ds,
                         cfg.data.imgs_per_gpu,
                         1,
                         dist=False) for ds in dataset
    ]

    # build runner
    optimizer = build_optimizer(cfg.optimizer)
    runner = Runner(model,
                    batch_processor,
                    optimizer,
                    cfg.work_dir,
                    logger=logger)
    # workaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    optimizer_config = cfg.optimizer_config
    optimizer_config['amp_enabled'] = mixed_precision
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(tf_datasets, cfg.workflow, cfg.total_epochs)
