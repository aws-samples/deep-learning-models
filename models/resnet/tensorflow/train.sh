# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Ensure you have Horovod, OpenMPI, and Tensorflow installed on each machine
# Specify hosts in the file `hosts`

if [ -z "$1" ]
  then
    gpus=64
  else
    gpus=$1
fi

echo "Launching training job using $gpus GPUs"
set -ex

# Training
# adjust the learning rate based on how many gpus are being used. 
# for x gpus use x*0.1 as the learning rate for resnet50 with 256batch size per gpu
mpirun -np $gpus -hostfile hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 \
	--lr_decay_mode poly --warmup_epochs 10 --clear_log

# Evaluation
# Using only 8 gpus for evaluation as we saved checkpoints only on master node
# pass num_gpus it was trained on to print the epoch numbers correctly
mpirun -np 8 -hostfile hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 \
	--eval --num_gpus $gpus