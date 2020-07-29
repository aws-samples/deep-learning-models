# Training scripts for common backbones used in Object Detection

## Instructions to train

### Data Setup

### EC2 training

mpirun -np 8 -H localhost:8 -map-by slot -x NCCL_DEBUG=INFO -x TF_XLA_FLAGS=--tf_xla_cpu_global_jit -mca btl ^openib -mca btl_tcp_if_exclude tun0,docker0,lo --bind-to none --allow-run-as-root python train_backbone.py --train_data_dir /data/imagenet/tf_records/train/ --validation_data_dir /data/imagenet/tf_records/validation -b 128 --model hrnet_w32c --schedule cosine

### SageMaker training
Coming soon


## Details of training

For most cases we use cosine decay scheduler and train for 120 epochs on Imagenet dataset.

Standard imagenet data augmentation techniques are used, in addition we use mixup to achieve improved results.


## Top-1 Imagenet accuracy 


| Num_GPUs x Images_Per_GPU | Instance type | Model | Top-1 Acc | Notes |
| ------------------------- | ------------- | ------------: | ------: | ----- |
| (1x8)x128 | P3.16xl | ResNet50V1_b | xx.yy% |  |
| (1x8)x128 | P3.16xl | ResNet50V1_d | xx.yy% |  |
| (1x8)x128 | P3.16xl | ResNet101V1_b | xx.yy% |  |
| (1x8)x128 | P3.16xl | ResNet101V1_d | xx.yy% |  |
| (1x8)x128 | P3.16xl | HRNetW32_c | xx.yy% |  |
| (1x8)x128 | P3.16xl | DarkNet53 | xx.yy% |  |

