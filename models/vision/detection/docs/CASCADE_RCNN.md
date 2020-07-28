# Cascade RCNN

TensorFlow 2.x based Cascade RCNN implementation using Feature Pyramid Networks and ResNet50 backbone

The original paper: [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726)

### Overview

This implementation of Cascade RCNN is focused on increasing training throughput without sacrificing accuracy. 

The implementation achieves fast training times through usage of multi image batches per GPU, mixed precision training, and TensorFlow autograph feature. The code is fully TF 2.x compatible and supports debugging in Eager mode as well.

Additionally, the backbone is easily swappable, and TF Keras pretrained weights can be used for initializing the backbone weights.

### Status

Training on N GPUs (V100s in our experiments) with a per-gpu batch size of M = NxM training


### Notes

- Running this codebase does not require any custom op modifications and achieves good training efficiency

### To launch training

- Data preprocessing (needs to be done only for EC2 training - SM training takes care of dataset setup for COCO)
  - We are using COCO 2017, you can download the data from [COCO data](http://cocodataset.org/#download).
  - The file folder needs to have the following directory structure:
  ```
  data/
    annotations/
      instances_train2017.json
      instances_val2017.json
    pretrained-models/
      ImageNet-R50-AlignPadding.npz
    train2017/
      # image files that are mentioned in the corresponding json
    val2017/
      # image files that are mentioned in corresponding json
  ```
  
  
  - EC2 Setup (single node training - instructions for multinode coming soon)
  
    Adjust configuration for your model and use the training script as follows:
      ```
      scripts/train_ec2_single_node.sh <NUM_GPUs> <path to config>
      ```


  - SageMaker Setup

    Follow AWSDet tutorial for your model at [AWSDet Tutorial](../tutorial/awsdet/Tutorial.ipynb)


### Training results


### Training results
The results were obtained on SageMaker (distributed training does not use EFA.) Training times do not include the time to setup data/model or running evaluation at the end of every epoch.
12 epochs training:

| Num_GPUs x Images_Per_GPU | Instance type | Training time | Box mAP | Notes |
| ------------------------- | ------------- | ------------: | ------: | ----- |
| 8x4 | P3dn.24xl | 7h 20m | 38.90% | |
| 16x4 | P3dn.24xl | 4h 20m | 38.60% | |
| 32x4 | P3dn.24xl | 2h 49m | 38.60% | |
| 64x2 | P3dn.24xl | 2h 36m | 38.30% | there are issues with batch size 2 and scaling that are currently being investigated |


### Attribution

The code is heavily inspired by the excellent MMDetection toolbox [Open MMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)