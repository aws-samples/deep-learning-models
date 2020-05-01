# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
NUM_GPU=${1:-1}
TRAIN_CFG=$2

echo ""
echo "NUM_GPU: ${NUM_GPU}"
echo "TRAIN_CFG: ${TRAIN_CFG}"
echo ""

cd /deep-learning-models/models/vision/detection
export PYTHONPATH=${PYTHONPATH}:${PWD}

mpirun -np ${NUM_GPU} \
--H localhost:${NUM_GPU} \
--allow-run-as-root \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_exclude lo,docker0 \
-mca btl_vader_single_copy_mechanism none \
-x LD_LIBRARY_PATH \
-x PATH \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_MIN_NRINGS=8 \
-x NCCL_DEBUG=INFO \
-x TF_CUDNN_USE_AUTOTUNE=0 \
-x HOROVOD_CYCLE_TIME=0.5 \
-x HOROVOD_FUSION_THRESHOLD=67108864 \
--output-filename /logs/mpirun_logs \
python tools/train.py ${TRAIN_CFG} \
--validate \
--autoscale-lr \
--amp

