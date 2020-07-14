# Faster RCNN/Mask RCNN

TensorFlow 2.x based Faster RCNN implementation using Feature Pyramid Networks and ResNet50 backbone

The original paper: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

### Overview

This implementation of Faster RCNN is focused on increasing training throughput without sacrificing accuracy. 

The implementation achieves fast training times through usage of multi image batches per GPU, mixed precision training, and TensorFlow autograph feature. The code is fully TF 2.x compatible and supports debugging in Eager mode as well.

Additionally, the backbone is easily swappable, and TF Keras pretrained weights can be used for initializing the backbone weights.

Our implementation provides an optional mask head, which can be enables by supplying a mask head configuration in the configuration file. An example default configuration can be found in the configuration files.

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
The results were obtained on SageMaker.
12 epochs training:

| Num_GPUs x Images_Per_GPU | Instance type | Training time | Box mAP | Seg mAP | Notes |
| ------------------------- | ------------- | ------------: | ------: | ------: | ----- |
| 8x4 | P3dn.24xl | 5h 15 | 36.40% | - | |
| 16x4 | P3dn.24xl | 2h 50 | 36.30% | - | |
| 32x4 | P3dn.24xl | 1h 27 | 35.70% | - | HP search not done |
| 64x2 | P3dn.24xl | 0h 56 | 35.50% | - | HP search not done |


### Example output
[TODO]

### Attribution

The code is heavily inspired by the excellent MMDetection toolbox [Open MMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)

Some parts of code have been borrowed or derived implementations from the following repositories
- [Viredery/tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn)
- [irvingzhang0512/tf_eager_object_detection](https://github.com/irvingzhang0512/tf_eager_object_detection)
- [TensorPack](https://github.com/tensorpack/tensorpack/)
