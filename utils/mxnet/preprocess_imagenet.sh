# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Download ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar from imagenet website

$TRAIN_DIR = train_data
$VAL_DIR = val_data
mkdir -p $TRAIN_DIR
mkdir -p $VAL_DIR

tar -xvf ILSVRC2012_img_train.tar -C $TRAIN_DIR
for i in $TRAIN_DIR/*.tar; do j=${i%.*}; echo $j;  mkdir -p $j; tar -xf $i -C $j; done
rm $TRAIN_DIR/*.tar

tar -xvf ILSVRC2012_img_val.tar -C $VAL_DIR

# Download `.lst` files
wget http://data.mxnet.io/models/imagenet/resnet/train.lst
wget http://data.mxnet.io/models/imagenet/resnet/val.lst

# Download `im2rec.py`
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py

# Execute, modify path to imagenet directory
python im2rec.py train ./$TRAIN_DIR --recursive --resize 480 --num-thread 30 --pack-label
python im2rec.py val ./$VAL_DIR --recursive --resize 480 --num-thread 16 --pack-label --no-shuffle