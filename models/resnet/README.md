# ResNet 
We provide implementations of training ResNet on the Imagenet dataset using MXNet and Tensorflow. 

The bash scripts provided here are configured to train Resnet50 on the Imagenet dataset to beyond 75.5% top 1 validation accuracy using a standard 90 epochs training schedule using 8 p3.16xlarge nodes. We use a batch size of 256 per GPU making the aggregate batch size to be 16k.