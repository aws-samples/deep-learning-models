# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Preprocess imagenet raw data to create TFRecords
# Need to specify your imagenet username and access key
python3 preprocess_imagenet.py --local_scratch_dir=./imagenet_data \
	--imagenet_username=[your imagenet account] --imagenet_access_key=[your imagenet access key]

# Resize training & validation data, maintaining the aspect ratio, 
python3 tensorflow_image_resizer.py -d imagenet -i [PATH TO TFRECORD TRAINING DATASET]  -o  [PATH TO RESIZED TFRECORD TRAINING DATASET] \
	--subset_name train --num_preprocess_threads 60 --num_intra_threads 2 --num_inter_threads 2
python3 tensorflow_image_resizer.py -d imagenet -i [PATH TO TFRECORD VALIDATION DATASET]  -o  [PATH TO RESIZED TFRECORD VALIDATION DATASET] \
	--subset_name validation --num_preprocess_threads 60 --num_intra_threads 2 --num_inter_threads 2
