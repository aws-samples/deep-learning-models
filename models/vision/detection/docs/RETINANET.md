# RetinaNet

TensorFlow 2.x based RetinaNet implementation using Feature Pyramid Networks and ResNet50 backbone

The original paper: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

### Overview

The implementation achieves fast training times through usage of multi image batches per GPU, mixed precision training, and TensorFlow autograph feature. The code is fully TF 2.x compatible and supports debugging in Eager mode as well.

For details on model configuration and setup please see [configs/retinanet](../configs/retinanet)

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

The results were obtained on SageMaker (distributed training does not use EFA.) Training times include the time to setup data/model as well as running evaluation at end of every epoch.

12 epochs (COCO 2017 validation 1x) single scale training:

| Num_GPUs x Images_Per_GPU | Instance type | Training time per epoch | Box mAP | Notes |
| ------------------------- | ------------- | ------------: | ------: | ----- |
| (1x8)x2 | P3.16xl | 32m | 35.40% |  |
| (1x8)x4 | P3.16xl | 28m | 35.40% |  |
| (2x8)x4 | P3dn.24xl | 17m | 34.80% |  |
| (4x8)x4 | P3dn.24xl | 11m | 34.80% |  |
| (8x8)x4 | P3dn.24xl | 7m | 34.40% | 1000 warmup iters |


### Known Issues
- Results are not deterministic - you can expect a delta of +/- 0.2% in mAP scores for training runs with the same settings
- Not much hyperparameter tuning has been done, you may obtain better results with hyperparameter search and multiscale training


### Attribution
The code is heavily inspired by the excellent MMDetection toolbox [Open MMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)
