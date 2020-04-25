# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Specify hosts in the file `hosts`, ensure that the number of slots is equal to the number of GPUs on that host

# Use train_more_aug.sh when training with large number of GPUs (128, 256, etc). That script uses more augmentations and layer wise adaptive rate control (LARC) to help with convergence at large batch sizes. 

# This script has been tested on DLAMI v17 and above

if [ -z "$1" ]
  then
    echo "Usage: "$0" <num_gpus>"
    exit 1
  else
    gpus=$1
fi

function runclust(){ while read -u 10 host; do host=${host%% slots*}; if [ ""$3"" == "verbose" ]; then echo "On $host"; fi; ssh -o "StrictHostKeyChecking no" $host ""$2""; done 10<$1; };

# Activating tensorflow_p36 on each machine
runclust hosts "echo 'Activating tensorflow_p36'; tmux new-session -s activation_tf -d \"source activate tensorflow_p36 > activation_log.txt;\"" verbose; 
# Waiting for activation to finish
runclust hosts "while tmux has-session -t activation_tf 2>/dev/null; do :; done; cat activation_log.txt"
# You can comment out the above two runclust commands if you have activated the environment on all machines at least once

# Activate locally for the mpirun command to use
source activate tensorflow_p36

echo "Launching training job using $gpus GPUs"
set -ex

# use ens3 interface for DLAMI Ubuntu and eth0 interface for DLAMI AmazonLinux
if [  -n "$(uname -a | grep Ubuntu)" ]; then INTERFACE=ens3 ; else INTERFACE=eth0; fi
NUM_GPUS_MASTER=`nvidia-smi -L | wc -l`

# p3 instances have larger GPU memory, so a higher batch size can be used
GPU_MEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0 | awk '{print $1}'`
if [ $GPU_MEM -gt 15000 ] ; then BATCH_SIZE=256; else BATCH_SIZE=128; fi

# Training
~/anaconda3/envs/tensorflow_p36/bin/mpirun -np $gpus -hostfile hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_exclude lo,docker0 \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 -b $BATCH_SIZE \
	--lr_decay_mode poly --warmup_epochs 10 --clear_log

# Evaluation
# Using only master node for evaluation as we saved checkpoints only on master node
# pass num_gpus it was trained on to print the epoch numbers correctly
~/anaconda3/envs/tensorflow_p36/bin/mpirun -np $NUM_GPUS_MASTER -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_exclude lo,docker0 \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 -b $BATCH_SIZE \
	--eval --num_gpus $gpus
