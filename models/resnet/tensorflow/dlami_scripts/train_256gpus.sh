# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Ensure you have Horovod, OpenMPI, and Tensorflow installed on each machine
# Specify hosts in the file `hosts`

# Below was tested on DLAMI v15 Ubuntu. 
# If you are using AmazonLinux change ens3 in line 15 to eth0
# If you have version 12 or older, you need to remove line 15 below.
source activate tensorflow_p36
cd ..
/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/mpirun -np 256 -hostfile hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x NCCL_SOCKET_IFNAME=ens3 -mca btl_tcp_if_exclude lo,docker0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--model resnet50 --fp16 --num_epochs 90 --warmup_epochs 5 --adv_bn_init --display_every 100 --loss_scale 256. --data_dir imagenet_data --lr 3.7 --log_dir results --brightness 0.3 --contrast 0.6 --saturation 0.6 --hue 0.13 --mom 0.977 --wdecay 0.0005 --batch_size 256 --adv_conv_init --lr_decay_mode 'linear_cosine' --eval_interval 10 --leta 0.012 --save_checkpoints_steps 190
