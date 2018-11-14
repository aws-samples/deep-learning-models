# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Ensure you have Horovod, OpenMPI, and Tensorflow installed on each machine
# Specify hosts in the file `hosts`

# Below was tested on DLAMI v15 Ubuntu.
# If you are using AmazonLinux change ens3 in line 15 to eth0
# If you have version 12 or older, you need to remove line 15 below.

set -ex

# set your learning rate = 0.1 * gpus
lr=3.2

if [ -z "$1" ]
  then
    gpus=8
  else
    gpus=$1
    echo "User is asking to run $gpus GPUs!"
fi

echo "Starting multi-node training process. First, let's clean things up..."
echo "Cleaning up all current python processes..."
export COMMAND="pkill -9 python"
while read -u 10 host
   do host=${host%% slots*}
   ssh -o "StrictHostKeyChecking no" $host "$COMMAND"
done
10<hosts
echo "Deleting logs folder..."
export COMMAND="rm -rf ~/resnet50_log/"; while read -u 10 host; do host=${host%% slots*}; ssh -o "StrictHostKeyChecking no" $host "$COMMAND"; done 10<hosts
echo "Checking disk space..."
export COMMAND="df /"; while read -u 10 host; do host=${host%% slots*}; ssh -o "StrictHostKeyChecking no" $host "$COMMAND"; done 10<hosts

source activate tensorflow_p36

# adjust the learning rate based on how many gpus are being used.
# for x gpus use x*0.1 as the learning rate for resnet50 with 256batch size per gpu
/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/mpirun -np $gpus -hostfile ~/hosts -mca plm_rsh_no_tree_spawn 1 \
        -bind-to socket -map-by slot \
        -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
        -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
        -x NCCL_SOCKET_IFNAME=ens3 -mca btl_tcp_if_exclude lo,docker0 \
        python -W ignore ~/deep-learning-models/models/resnet/tensorflow/train_imagenet_resnet_hvd.py \
        --model resnet50 --fp16 --num_epochs 90 --warmup_epochs 10 --adv_bn_init \
        --lr $lr --lr_decay_mode poly \
        --data_dir ~/data/tf-imagenet/ --log_dir ~/resnet50_log

# Using only 8 gpus for evaluation as we saved checkpoints only on master node
# pass num_gpus it was trained on to print the epoch numbers correctly
/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/mpirun -np 8 -hostfile ~/hosts -mca plm_rsh_no_tree_spawn 1 \
        -bind-to socket -map-by slot \
        -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
        -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
        -x NCCL_SOCKET_IFNAME=ens3 -mca btl_tcp_if_exclude lo,docker0 \
        python -W ignore ~/deep-learning-models/models/resnet/tensorflow/train_imagenet_resnet_hvd.py \
        --model resnet50 --fp16 --adv_bn_init --num_epochs 2 --num_gpus 16 \
        --data_dir ~/data/tf-imagenet/ --log_dir ~/resnet50_log --eval
