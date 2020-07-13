# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
import tensorflow as tf
from s3fs import S3FileSystem
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from awsdet.utils.runner.dist_utils import master_only
from .base import LoggerHook


class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 s3_dir=None,
                 interval=10,
                 image_interval=None,
                 s3_interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 diagnostic=False,
                 diagnostic_interval=500):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.log_dir = log_dir
        if log_dir is None and not (s3_dir is None):
            # SageMaker setup
            self.log_dir = Path(os.getenv('SM_OUTPUT_DATA_DIR')).as_posix()
        self.s3_dir = s3_dir
        self.s3_interval = s3_interval
        self.image_interval = image_interval
        self.diagnostic = diagnostic
        self.diagnostic_interval = diagnostic_interval
        if s3_dir:
            self.s3 = S3FileSystem()
            self.threadpool = ThreadPoolExecutor()

    @master_only
    def before_run(self, runner):
        self.writer = tf.summary.create_file_writer(self.log_dir)

    @master_only
    def log(self, runner):
        with self.writer.as_default():
            for var in runner.log_buffer.output:
                if var in ['time', 'data_time']:
                    continue
                tag = '{}/{}'.format(var, runner.mode)
                record = runner.log_buffer.output[var]
                if isinstance(record, str):
                    tf.summary.text(tag, record, step=runner.iter)
                else:
                    tf.summary.scalar(tag, record, step=runner.iter)
            self._image_log(runner)
            if self.diagnostic:
                self._model_diagnostics(runner)
        self.writer.flush()
        self._s3_upload(runner)
        
    
    def _model_diagnostics(self, runner):
        grads = runner.grads
        weights = runner.model.trainable_variables
        var_means = {"var_mean/{}".format(i.name):tf.reduce_mean(i).numpy() \
                     for i in runner.model.trainable_variables}
        var_std = {"var_std/{}".format(i.name):tf.math.reduce_std(i).numpy() \
                   for i in runner.model.trainable_variables}
        var_mag = {"var_mag/{}".format(i.name):tf.reduce_sum(tf.square(i)).numpy() \
                   for i in runner.model.trainable_variables}
        grad_means = {"grad_means/{}".format(i.name):tf.reduce_mean(j).numpy() \
                  for i,j in zip(runner.model.trainable_variables, 
                                 runner.grads)}
        grad_std = {"grad_std/{}".format(i.name):tf.math.reduce_std(j).numpy() \
                  for i,j in zip(runner.model.trainable_variables, 
                                 runner.grads)}
        grad_mag = {"grad_mag/{}".format(i.name):tf.reduce_sum(tf.square(j)).numpy() \
                  for i,j in zip(runner.model.trainable_variables, 
                                 runner.grads)}
        grad_max = {"grad_max/{}".format(i.name):tf.math.reduce_max(j).numpy() \
                  for i,j in zip(runner.model.trainable_variables, 
                                 runner.grads)}
        grad_norm = {"grad_norm/{}".format(i.name):tf.norm(j).numpy() \
                  for i,j in zip(runner.model.trainable_variables, 
                                 runner.grads)}
        grad_hist = {"grad_hist/{}".format(i.name):j.numpy() \
                    for i,j in zip(runner.model.trainable_variables, runner.grads)}
        var_hist = {"var_hist/{}".format(i.name):i.numpy() \
                    for i in runner.model.trainable_variables}
        for i,j in var_means.items():
            tf.summary.scalar(i, j, step=runner.iter)
        for i,j in var_std.items():
            tf.summary.scalar(i, j, step=runner.iter)
        for i,j in var_mag.items():
            tf.summary.scalar(i, j, step=runner.iter)
        for i,j in grad_means.items():
            tf.summary.scalar(i, j, step=runner.iter)
        for i,j in grad_std.items():
            tf.summary.scalar(i, j, step=runner.iter)
        for i,j in grad_mag.items():
            tf.summary.scalar(i, j, step=runner.iter)
        for i,j in grad_max.items():
            tf.summary.scalar(i, j, step=runner.iter)
        for i,j in grad_norm.items():
            tf.summary.scalar(i, j, step=runner.iter)
        if self.every_n_inner_iters(runner, self.image_interval+self.diagnostic_interval):
            for i,j in grad_hist.items():
                tf.summary.histogram(i, j, step=runner.iter)
            for i,j in var_hist.items():
                tf.summary.histogram(i, j, step=runner.iter)
        
    def _image_log(self, runner):
        if self.image_interval and \
        self.every_n_inner_iters(runner, self.image_interval+self.interval):
            for var in runner.log_buffer.val_history:
                if 'image' in var:
                    tag = '{}/{}'.format(runner.mode, var)
                    record = runner.log_buffer.val_history[var][-1]
                    tf.summary.image(tag, record, step=runner.iter+1)
                    
    def _s3_upload(self, runner):
        if self.s3_dir and \
        self.every_n_inner_iters(runner, self.s3_interval):
            event_file = list(Path(self.log_dir).glob("events*"))
            for file in event_file:
                # self.s3.put(file, Path(self.s3_dir).joinpath(file.name))
                _ = self.threadpool.submit(self.s3.put, file, 
                                           Path(self.s3_dir).joinpath(file.name))
        
    @master_only
    def after_run(self, runner):
        self.writer.close()
