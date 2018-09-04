# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Ensure you have Horovod, OpenMPI, and Tensorflow installed on each machine
# Specify hosts in the file `hosts`
mpirun -np 64 -hostfile hosts -mca plm_rsh_no_tree_spawn 1 -bind-to socket -map-by slot -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python -W ignore train_imagenet_resnet_hvd.py --model resnet50 --fp16 --num_epochs 90 --warmup_epochs 10 --adv_bn_init --display_every 100 --lr 6.4 --loss_scale 1024. --lr_decay_mode poly --data_dir imagenet_data --log_dir resnet50_log
