# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# PreRequisities
# Install mxnet >=1.3b and gluoncv on each machine

# Clone source to get launch.py script to start training job
git clone --recursive https://github.com/apache/incubator-mxnet

# Ensure data is kept at ~/data for each machine or change the data paths below
# Example command to start the training job
# Specify hosts in the file `hosts`
incubator-mxnet/tools/launch.py -n 8 -H hosts python train_imagenet.py --use-rec --batch-size 256 --dtype float16 --num-data-workers 40 --num-epochs 90 --gpus 0,1,2,3,4,5,6,7 --lr 6.4 --lr-mode poly --warmup-epochs 10 --last-gamma --mode symbolic --model resnet50_v1b --kvstore dist_sync_device --rec-train ~/data/train.rec --rec-train-idx ~/data/train.idx --rec-val ~/data/val.rec --rec-val-idx ~/data/val.idx --save-frequency 1000 --warmup-lr 0.001 2>&1 | tee resnet50.log
