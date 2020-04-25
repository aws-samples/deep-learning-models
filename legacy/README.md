## Deep Learning Models

This repository contains scripts to train deep learning models optimized to run well on AWS. Apart from scripts to build and train the model, we also share here scripts to setup a high performing cluster for deep learning using AWS, and preprocessing scripts to prepare datasets.

Currently, it has scripts for training Resnet50 with Imagenet using Apache MXNet and Tensorflow.
Feel free to create a Github issue here if you have any questions.

### Manual cluster setup
- Ensure that the security group of the instances allows connections through any port from within the same security group.
- Ensure that your instances in the cluster have passwordless ssh set up. You should be able to do `ssh IP1` where IP1 can be the IP of any node in the cluster, from any node in the cluster. One easy way to do this would be to use Agent Forwarding. Here is how to enable that.
```
eval `ssh-agent`
ssh-add key.pem
ssh -A MASTER_NODE
``` 

### Running on AWS EC2 Deep Learning AMI
Make sure to use the `train_dlami.sh` script which handles the docker interface and conda environments.

### Production ready cluster setup
Check out `hpc-cluster` in the repository which sets up a high performance cluster for deep learning. It uses best practices such as bastion hosts and a BeeGFS distributed file system across all nodes in the cluster for high performant store for the dataset.

## License Summary

This sample code is made available under a modified MIT license. See the LICENSE file.
