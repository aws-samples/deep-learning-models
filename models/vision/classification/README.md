# Training scripts for common backbones used in Object Detection
  
The script trains on ImageNet dataset from scratch. Fine-tuning is not supported at the moment. Trained backbones can be used in the object detection library under `detection` directory.


## Instructions to train

### Data Setup

Download ImageNet data and prepare TF Records according to the script.
Please see script [here](https://github.com/aws-samples/deep-learning-models/blob/master/legacy/utils/tensorflow/preprocess_imagenet.py)

### Docker Image

Prepare a docker image for training. A sample Dockerfile is available [here](https://github.com/aws-samples/deep-learning-models/blob/master/models/vision/detection/docker/Dockerfile.ec2)

### EC2 training

Inside the docker container

```
# Train a HRNet_W32C classifier

$ mpirun -np 8 -H localhost:8 -map-by slot -x NCCL_DEBUG=INFO -x TF_XLA_FLAGS=--tf_xla_cpu_global_jit -mca btl ^vader -mca btl_tcp_if_exclude tun0,docker0,lo --bind-to none --allow-run-as-root python train_backbone.py --train_data_dir /data/imagenet/tf_records/train/ --validation_data_dir /data/imagenet/tf_records/validation -b 128 --model hrnet_w32c --schedule cosine
```

### SageMaker training

WIP


## Details of training

For most cases we use cosine decay scheduler and train for 120 epochs on ImageNet dataset.

Standard imagenet data augmentation techniques are used, in addition we use [mixup](https://arxiv.org/abs/1710.09412) and label smoothing to achieve improved results.


## Top-1 Imagenet accuracy 


| Num_GPUs x Images_Per_GPU | Instance type | Model | Top-1 Acc | Training Notes |
| ------------------------- | ------------- | ------------: | ------: | ----- |
| (1x8)x128 | P3.16xl | ResNet50V1_b | 76.8 |  7.3 iters/sec |
| (1x8)x128 | P3.16xl | ResNet50V1_d | 77.6 |  6.2 iters/sec|
| (1x8)x128 | P3.16xl | ResNet101V1_b | 78.9 | 4.8 iters/sec |
| (1x8)x128 | P3.16xl | ResNet101V1_d | 79.5 | 4.3 iters/sec |
| (1x8)x128 | P3.16xl | HRNetW32_c | 79.1 | 2.3 iters/sec |
| (1x8)x128 | P3.16xl | DarkNet53 | 77.0 | 6.4 iters/sec, crop dim 256 |

