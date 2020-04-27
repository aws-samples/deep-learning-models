# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile
import tensorflow as tf
from pycocotools.cocoeval import COCOeval
from awsdet.core import coco_eval, results2json
from awsdet.datasets import build_dataloader, build_dataset
from awsdet.models import build_detector
from awsdet.utils.misc import Config, mkdir_or_exist
from awsdet.utils.runner.dist_utils import get_dist_info, init_dist
from awsdet.utils import fileio
from awsdet.core.bbox import transforms

gpus = tf.config.experimental.list_physical_devices('GPU')
init_dist()

if not gpus:
    distributed = False  # single node single gpu
else:
    distributed = True

def load_checkpoint(model, filename):
    print('Loading checkpoint from %s...', filename)
    model.load_weights(filename)
    print('Loaded weights from checkpoint: {}'.format(filename))


def evaluate(dataset, results):
    tmp_file = 'temp_0'
    result_files = results2json(dataset, results, tmp_file)

    #res_types = ['bbox', 'segm'
    #            ] if runner.model.module.with_mask else ['bbox']
    res_types = ['bbox']
    cocoGt = dataset.coco
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
        print((
            '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
            '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6]))

def cpu_test(model, dataset, show=False):
    # create a loader for this runner
    tf_dataset, num_examples = build_dataloader(dataset, 1, 1, num_gpus=1, dist=False)
    # num_examples=8
    results = []

    for i, data_batch in enumerate(tf_dataset):
        if i >= num_examples:
            break
        _, img_meta = data_batch
        print(dataset.img_ids[i])
        outputs = model(data_batch, training=False)
        bboxes = outputs['bboxes']
        # # map boxes back to original scale
        bboxes = transforms.bbox_mapping_back(bboxes, img_meta)
        # # print('>>>>', bboxes)
        labels = outputs['labels']
        scores = outputs['scores']
        result = transforms.bbox2result(bboxes, labels, scores, num_classes=81)
        #for b, l, s in zip(bboxes, labels, scores):
        #    print(b, l, s)
        #print(result)
        results.append(result)
    evaluate(dataset, results)
    return results

def single_gpu_test(model, data_loader, show=False):
    raise NotImplementedError

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet TF test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    # parser.add_argument(
    #     '--gpu_collect',
    #     action='store_true',
    #     help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = Config.fromfile(args.config)

    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        raise NotImplementedError

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    # dummy data to init network
    img = tf.random.uniform(shape=[1333, 1333, 3], dtype=tf.float32)
    img_meta = tf.constant(
        [465., 640., 3., 800., 1101., 3., 1333., 1333., 3., 1.7204301, 0.],
        dtype=tf.float32)

    _ = model((tf.expand_dims(img, axis=0), tf.expand_dims(img_meta, axis=0)),
              training=False)

    load_checkpoint(model, args.checkpoint)
    model.CLASSES = dataset.CLASSES

    if not distributed:
        outputs = cpu_test(model, dataset, args.show)
    else:
        raise NotImplementedError

    rank, _, _, _ = get_dist_info()

    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        fileio.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out)
                    coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco)

    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    main()
