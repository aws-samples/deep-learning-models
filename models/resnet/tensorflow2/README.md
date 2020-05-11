## Resnet training example

The instructions below decribe how to train a resnet model on the AWS Deep Learning Container (DLC).

### Prerequisites 

In order to train this model, you need access to Imagenet data in TFRecord format.

Instructions for converting raw Imagenet data to TFRecord can be found [here](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py).

Your data should be stored on S3 in the format

```
s3://my-bucket/imagenet/train-00001-of-01024
s3://my-bucket/imagenet/train-00002-of-01024
.
.
.
s3://my-bucket/imagenet/validation-00128-of-00128
```

### Instance setup

On your EC2 instance copy your data from S3

```
mkdir -p ~/shared_workspace/imagenet/data
aws s3 cp --recursive s3://my-bucket/imagenet ~/shared_workspace/imagenet/data
```

Pull the DLC from AWS ECR

```
ECR_REPO=763104351884.dkr.ecr.us-east-1.amazonaws.com
DLC=tensorflow-training:2.1.0-gpu-py36-cu101-ubuntu18.04

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ECR_REPO}
docker pull ${ECR_REPO}/${DLC}
```

Clone this repo

```git clone -b tf2 https://github.com/aws-samples/deep-learning-models/ ~/shared_workspace/imagenet/deep-learning-models```

Create a hostfile for MPI. Note that for a single instance, you can copy the command below. For training across multiple EC2 instances, this file should contain the private ip addresses of all instances.

```
GPU_COUNT=`expr \`nvidia-smi --query-gpu=gpu_name --format=csv | wc -l\` - 1`
printf "localhost\tslots=${GPU_COUNT}\n" >> ~/shared_workspace/hosts
```

### Launch training with DLC

The Resnet script runs with MPI. Below is an example command to launch training.

```
RESNET_TRAIN="mpirun --hostfile /root/shared_workspace/hosts \
                     -mca plm_rsh_no_tree_spawn 1 \
                     -bind-to socket -map-by slot \
                     -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
                     -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
                     -x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_exclude lo,docker0 \
                     -x TF_CPP_MIN_LOG_LEVEL=0 \
                     python -W ignore /root/shared_workspace/imagenet/deep-learning-models/models/resnet/tensorflow2/train_tf2_resnet.py \
                     --data_dir /root/shared_workspace/imagenet/data"
```

This can be run interactively in the container, or launched with docker run.

```
docker run --rm -it --gpus all \
    --name tf2-resnet \
    --net=host --uts=host --ipc=host \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    -v ~/shared_workspace:/root/shared_workspace \
    ${ECR_REPO}/${DLC} /bin/bash -c "${RESNET_TRAIN}"
```