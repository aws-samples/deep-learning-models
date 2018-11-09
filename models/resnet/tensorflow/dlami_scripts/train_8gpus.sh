# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Ensure you have Horovod, OpenMPI, and Tensorflow installed on each machine
# Specify hosts in the file `hosts`

# Below was tested on DLAMI v17 Ubuntu. 
# If you are using AmazonLinux change ens3 in line 20 to eth0
# If you have version 12 or older, you need to remove line 20 below.

source activate tensorflow_p36
cd ..
# Training
/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/mpirun -np 8 -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=ens3 -mca btl_tcp_if_exclude lo,docker0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--lr 0.8 --lr_decay_mode poly \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90
# If you want to run with dummy data instead, replace last line with the following: --synthetic --num_batches 1000

# Evaluation
/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/mpirun -np 8 -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=ens3 -mca btl_tcp_if_exclude lo,docker0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 --eval