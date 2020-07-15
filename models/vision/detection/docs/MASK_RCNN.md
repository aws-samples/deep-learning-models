# Mask RCNN

Tensorflow 2.x based Mask RCNN using Feature Pyramid and ResNet50 backbone

Original paper: [Mask R-CNN](https://arxiv.org/abs/1703.06870)

### Overview

This Mask RCNN builds on our implementation of Faster RCNN to provide high training throughput, while maintaining accuracy and code extensibility.

The implementation uses multi images batches per GPU, mixed precision training, and Tensorflow autograph, as well as eager mode.

### Status

Training on N GPUs (V100s in our experiments) with a per-gpu batch size of M = NxM training

### Notes

- This codebase has been tested on Tensorflow 2.1, and achieves good training performance without custom ops.

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
The results were obtained on SageMaker.
12 epochs training:

| Num_GPUs x Images_Per_GPU | Instance type | Training time | Box mAP | Box mAP | Notes |
| ------------------------- | ------------- | ------------: | ------: | ------: | ----- |
| 8x4 | P3dn.24xl | 8h 05 | 37.10% | 32.20% | |

### Example output
[TODO]

### Attribution

The code is heavily inspired by the excellent MMDetection toolbox [Open MMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)

Some parts of code have been borrowed or derived implementations from the following repositories
- [Viredery/tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn)
- [irvingzhang0512/tf_eager_object_detection](https://github.com/irvingzhang0512/tf_eager_object_detection)
- [TensorPack](https://github.com/tensorpack/tensorpack/)