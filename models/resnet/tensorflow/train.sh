# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Ensure you have Horovod, OpenMPI, and Tensorflow installed on each machine
# Specify hosts in the file `hosts`

if [ -z "$1" ]
  then
    echo "Usage: "$0" <num_gpus>"
    exit 1
  else
    gpus=$1
fi

echo "Launching training job using $gpus GPUs"
set -ex

# p3 instances have larger GPU memory, so a higher batch size can be used
GPU_MEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0 | awk '{print $1}'`
if [ $GPU_MEM -gt 15000 ] ; then BATCH_SIZE=256; else BATCH_SIZE=128; fi

# Training
mpirun -np $gpus -hostfile hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 -b $BATCH_SIZE \
	--lr_decay_mode poly --warmup_epochs 10 --clear_log

NUM_GPUS_MASTER=`nvidia-smi -L | wc -l`

# Evaluation
# Using only the gpus on master node for evaluation as we saved checkpoints only on master node
# pass num_gpus it was trained on to print the epoch numbers correctly
mpirun -np $NUM_GPUS_MASTER -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 \
	--eval --num_gpus $gpus