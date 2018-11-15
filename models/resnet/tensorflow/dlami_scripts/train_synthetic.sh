# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Ensure you have Horovod, OpenMPI, and Tensorflow installed on each machine
# Specify hosts in the file `hosts`

# Below was tested on DLAMI v17 Ubuntu. 
# If you are using AmazonLinux change ens3 to eth0
# If you have version 12 or older, you need to remove the NCCL_SOCKET_IFNAME and -mca btl_tcp_if_exclude lo,docker0

if [ -z "$1" ]
  then
    gpus=64
  else
    gpus=$1
fi

function runclust(){ while read -u 10 host; do host=${host%% slots*}; ssh -o "StrictHostKeyChecking no" $host ""$2""; done 10<$1; };
runclust hosts "source activate tensorflow_p36 &"

echo "Launching training job with synthetic data using $gpus GPUs"
set -ex

# Training
# adjust the learning rate based on how many gpus are being used. 
# for x gpus use x*0.1 as the learning rate for resnet50 with 256batch size per gpu
/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/mpirun -np $gpus -hostfile hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=ens3 -mca btl_tcp_if_exclude lo,docker0 \
	python -W ignore ~/deep-learning-models/models/resnet/tensorflow/train_imagenet_resnet_hvd.py \
	--synthetic --num_epochs 5 --clear_log