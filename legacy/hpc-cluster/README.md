# Deep Learning High Performance Computing Cluster

We provide cloud formation scripts which can standup a High Performance Computing (HPC) cluster for deep learning as used by the work on `Scalable Multi-Node Deep Learning Training Using GPUs in the Cloud`. These instructions help setup a compute cluster for worker nodes and parameter stores and a BeeGFS RAM-based storage cluster on top of the compute cluster. Here's an illustration of this cluster.

![Deep Learning HPC Cluster](images/cluster.png)

Here are step wise instructions on using these scripts to set up this cluster.

1. Create keys for all the nodes in the cluster to use for communication. You can do this by generating a new set of keys 'ssh-keygen -t rsa'. Then create a folder called keys in this folder and copy both public and private keys here. 

2. Create an S3 bucket and upload the contents of this directory there and note the S3 bucket name. 

3. Create a VPC with a specified CIDR range. Ideally, you should create a public subnet and a private subnet with appropiate CIDR ranges to accommodate the number of requested IP addresses (number of nodes in the cluster). These ranges should be within the VPC. The private subnet will need to have routing into the public subnet via a NAT gateway. For details on setting this up go here: https://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/vpc-nat-gateway.html#nat-gateway-creating. Alternatively, you can create two public subnets if you are okay with allowing public IP adresses for your worker nodes.

4. Next, go to AWS Cloudformation choose the deeplearning-cluster.yml file given here and fill out the parameters as needed. An example of the parameters is shown below. Ensure your account has the instance limits you specify here. Creation of the stack can take a few minutes.
![Cloudformation template example](images/cloudformation-example.png)

5. Once the stack is up, you will have a bastion host to login to. The distributed file system is mounted at the ParallelMount point mentioned (/mnt/parallel) in the above image. Use this location to fetch your data and scripts, so you can start the job as per the framework's instructions.
