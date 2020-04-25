# Copyright (c) Open-MMLab. All rights reserved.
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
                 reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.log_dir = log_dir
        self.s3_dir = s3_dir
        self.s3_interval = s3_interval
        self.image_interval = image_interval
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
        self.writer.flush()
        self._s3_upload(runner)
    
    
    
    
    def _image_log(self, runner):
        if self.image_interval and \
        self.every_n_inner_iters(runner, self.image_interval+self.interval):
            for var in runner.log_buffer.val_history:
                if 'image' in var:
                    tag = '{}/{}'.format(var, runner.mode)
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
