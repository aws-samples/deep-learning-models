# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Specify hosts in the file `hosts`

# Below was tested on DLAMI v17 Ubuntu. 
# If you have version 12 or older, you need to remove the NCCL_SOCKET_IFNAME and -mca btl_tcp_if_exclude lo,docker0

if [ -z "$1" ]
  then
    gpus=256
  else
    gpus=$1
fi

function runclust(){ while read -u 10 host; do host=${host%% slots*}; if [ ""$3"" == "verbose" ]; then echo "On $host"; fi; ssh -o "StrictHostKeyChecking no" $host ""$2""; done 10<$1; };

# Activating tensorflow_p36 on each machine
runclust hosts "echo 'Activating tensorflow_p36'; tmux new-session -s activation_tf -d \"source activate tensorflow_p36 > activation_log.txt;\"" verbose; 
# Waiting for activation to finish
runclust hosts "while tmux has-session -t activation_tf 2>/dev/null; do :; done; cat activation_log.txt"
# Activate locally for the mpirun command to use
source activate tensorflow_p36

echo "Launching training job using $gpus GPUs"
set -ex

if [  -n "$(uname -a | grep Ubuntu)" ]; then INTERFACE=ens3 ; else INTERFACE=eth0; fi
NUM_GPUS_MASTER=`nvidia-smi -L | wc -l`

# Training
# This script is for training with large number of GPUs (large batch sizes). 
# You can for instance just replace the number of GPUs to 128 with the same script.
~/anaconda3/envs/tensorflow_p36/bin/mpirun -np $gpus -hostfile hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_exclude lo,docker0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 --increased_aug \
	--mom 0.977 --wdecay 0.0005 --loss_scale 256. --use_larc \
	--lr_decay_mode linear_cosine --warmup_epochs 5 --clear_log

# Evaluation
# Using only gpus on master node for evaluation as we saved checkpoints only on master node
# pass num_gpus it was trained on to print the epoch numbers correctly
~/anaconda3/envs/tensorflow_p36/bin/mpirun -np $NUM_GPUS_MASTER -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_exclude lo,docker0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 \
	--eval --num_gpus $gpus
