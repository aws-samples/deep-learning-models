# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import os
import os.path as osp
import numpy as np
import glob, time
from pycocotools.cocoeval import COCOeval
from awsdet.utils.runner import Hook
from awsdet.utils.runner.dist_utils import get_barrier, master_only
from awsdet import datasets
from .coco_utils import fast_eval_recall, results2json
from .mean_ap import eval_map
from awsdet.utils.misc import ProgressBar
from awsdet.datasets import build_dataloader
from awsdet.utils.fileio import load, dump
from awsdet.core.bbox import transforms
from awsdet.core.mask.transforms import mask2result
import tensorflow as tf

class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a dict from config, not {}'.format(
                    type(dataset)))
        self.interval = interval

    @master_only
    def _accumulate_results(self, runner, results, num_examples):
        # accumulate on the master
        for worker_idx in range(1, runner.local_size):
            worker_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(worker_idx))
            tmp_results = load(worker_file)
            for idx in range(num_examples):
                adjusted_idx = idx*runner.local_size+worker_idx
                results[adjusted_idx] = tmp_results[adjusted_idx]
            print('cleaning up', worker_file)
            os.remove(worker_file) # cleanup
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        # create a loader for this runner
        tf_dataset, num_examples = build_dataloader(self.dataset, 1, 1, num_gpus=runner.local_size, dist=True)
        # num_examples=8
        results = [None for _ in range(num_examples*runner.local_size)] # REVISIT - may require a lot of memory
        if runner.model.mask:
            masks = [None for _ in range(num_examples*runner.local_size)]
        if runner.rank == 0:
            prog_bar = ProgressBar(num_examples)
        for i, data_batch in enumerate(tf_dataset):
            if i >= num_examples:
                break
            _, img_meta = data_batch
            outputs = runner.model(data_batch, training=False)
            assert isinstance(outputs, dict)
            bboxes = outputs['bboxes']
            # map boxes back to original scale
            bboxes = transforms.bbox_mapping_back(bboxes, img_meta)
            labels = outputs['labels']
            scores = outputs['scores']
            result = transforms.bbox2result(bboxes, labels, scores, num_classes=self.dataset.CLASSES+1) # add background class
            if runner.model.mask:
                mask = mask2result(outputs['masks'], labels, img_meta[0])
                results[i*runner.local_size+runner.local_rank] = (result, mask)
            else:
                results[i*runner.local_size+runner.local_rank] = result
            if runner.rank == 0:
                prog_bar.update()
        # write to a file
        tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(runner.rank))
        if runner.rank != 0:
            dump(results, tmp_file)
            # open(tmp_file+'.done', 'w').close()
        # MPI barrier through horovod
        _ = get_barrier()
        self._accumulate_results(runner, results, num_examples)


    def evaluate(self, runner, results):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        annotations = [
            self.dataset.get_ann_info(i) for i in range(len(self.dataset))
        ]
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            annotations,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            logger=runner.logger)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 interval=1,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(
            dataset, interval=interval)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        res_types = ['bbox', 'segm'] if runner.model.mask else ['bbox']
        # res_types = ['bbox']
        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            try:
                cocoDt = cocoGt.loadRes(result_files[res_type])
            except IndexError:
                print('No prediction found.')
                break
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True

