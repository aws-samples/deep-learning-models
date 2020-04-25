# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os.path as osp
import time
import tensorflow as tf
from . import hooks
from .dist_utils import get_dist_info, get_distributed_tape, broadcast_weights
from .hooks import (CheckpointHook, Hook, IterTimerHook, LrUpdaterHook, 
                    lr_updater, OptimizerHook, WeightsMonitorHook)
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import get_host_info, get_time_str, obj_from_dict
from awsdet.utils.misc import mkdir_or_exist
from awsdet.utils.generic import is_list_of

import six

def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)

class Runner(object):
    """A training helper.

    Args:
        model (:obj:`tf.keras.Model`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`keras.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 amp_enabled=False):
        assert callable(batch_processor)
        self.model = model
        if optimizer is not None:
            self.optimizer = optimizer 
        else:
            self.optimizer = None
        self.batch_processor = batch_processor

        # create work_dir
        if is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        self._model_name = self.model.__class__.__name__

        self._rank, self._local_rank, self._world_size, self._local_size = get_dist_info()
        self.timestamp = get_time_str()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0
        self._amp_enabled = amp_enabled

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name


    @property
    def local_rank(self):
        """int: local rank of current process"""
        return self._local_rank


    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def local_size(self):
        """int: Number of processes running in the same node as this runner.
        (distributed training)"""
        return self._local_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = '{}.log'.format(self.timestamp)
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate (#TODO: support individual LR for param groups)
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return float(self.optimizer.learning_rate.numpy())

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename):
        self.logger.info('Loading checkpoint from %s...', filename)
        self.model.load_weights(filename)
        self.logger.info('Loaded weights from checkpoint: {}'.format(filename))

    def save_checkpoint(self, out_dir):
        filepath = osp.join(out_dir, self.model.name)
        # save full model, including optimizer state
        self.model.save_weights(filepath, save_format='tf')
        self.logger.info('Saved checkpoint at: {}'.format(filepath))

    @tf.function(experimental_relax_shapes=True)
    def run_train_step(self, data_batch):
        with tf.GradientTape() as tape:
            outputs = self.batch_processor(self.model, data_batch, train_mode=True)
        var_list = self.model.trainable_variables
        tape = get_distributed_tape(tape) if self.world_size > 1 else tape
        loss = outputs['loss']
        grads = tape.gradient(loss, var_list)
        grads = [
            grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)
        ]
        # all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in grads])
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, 15.0)
        # tf.print(global_norm, all_are_finite)
        self.optimizer.apply_gradients(zip(clipped_grads, var_list))
        return outputs

    def run_eval_step(self, data_batch):
        '''
        This exists only for purpose of debugging - check if model can predict train data that it may have seen
        Supports only one image at the moment
        '''
        if self.rank != 0:
            return
        imgs, img_metas, gt_boxes, gt_class_ids = data_batch
        detections_dict = self.batch_processor(self.model, (tf.expand_dims(imgs[0], axis=0), tf.expand_dims(img_metas[0], axis=0)), train_mode=False)
        for l, b in zip(gt_class_ids,gt_boxes):
            print('GT', l, b)
            print('DT:')
            for i in range(detections_dict['bboxes'].shape[0]):
                print(detections_dict['bboxes'][i], detections_dict['labels'][i], detections_dict['scores'][i])
            break # one image only


    def train(self, tf_dataset, **kwargs):
        self.mode = 'train'
        self.num_examples = tf_dataset[1]
        self._max_iters = self._max_epochs * self.num_examples
        self.broadcast = True
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(tf_dataset[0]):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.run_train_step(data_batch)
            if self.broadcast: # broadcast once
                broadcast_weights(self)
                self.broadcast = False
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if self.rank == 0 and 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
                # add current learning rate for tensorboard as well
                self.log_buffer.update({'learning_rate': self.current_lr()})
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1
            if i > 0 and i % 1000 == 0:
                self.run_eval_step(data_batch)
            if i+1 >= self.num_examples: # for case where num_examples is deliberately made small to test
                self._inner_iter = 0
                break

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, tf_dataset, **kwargs):
        raise NotImplementedError


    def resume(self, checkpoint):
        self.load_checkpoint(checkpoint)


    def run(self, tf_datasets, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`tf.data.datasets`]): Datasets for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(tf_datasets, list)
        assert is_list_of(workflow, tuple)
        assert len(tf_datasets) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow # ('train', 1)
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(tf_datasets[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        # self.register_hook(self.build_hook(optimizer_config, OptimizerHook), priority='VERY_HIGH')
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        # self.register_hook(WeightsMonitorHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)
