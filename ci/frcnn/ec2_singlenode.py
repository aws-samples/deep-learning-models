################################################################
# Launch DLAMI with EFA
################################################################

import boto3
import yaml
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--instance_id")
parser.add_argument("--docker_user")
parser.add_argument("--keypair")

args = parser.parse_args()
keypair = os.getcwd() + "/" + args.keypair

ec2_session = boto3.Session(region_name="us-east-1")
ec2_client = ec2_session.client("ec2")
ec2_resource = ec2_session.resource("ec2")

#response = ec2_client.run_instances(**config)
response = ec2_client.start_instances(InstanceIds=[args.instance_id])
print(response)
################################################################
# Create SSH interface to all instances
# Runs in loop while waiting for instances to be ready
################################################################
import ssh
from time import sleep

while True:
    try:
        instances = [instance['InstanceId'] for instance in response['StartingInstances']]
        status = ec2_resource.meta.client.describe_instances(InstanceIds=instances)
        public_ips = [instance['PublicIpAddress'] for instance in status['Reservations'][0]['Instances']]
        ssh_client = ssh.SSH(public_ips, keypair)
        # wait a few seconds and run a simple command to make sure instances are up
        pci = ssh_client.run_on_all('lspci')
        break
    except:
        sleep(10)
        continue
print(pci[0]['stdout'])


'''################################################################
# Setup Containers
################################################################

launch_cont = """docker run --rm -it -d --gpus all \
                    --name mpicont \
                    --net=host --uts=host --ipc=host \
                    --ulimit stack=67108864 --ulimit memlock=-1 \
                    --security-opt seccomp=unconfined \
                    -v /opt/amazon/efa:/efa \
                    -v /home/ubuntu/ssh_container:/root/.ssh \
                    -v ~/shared_workspace:/workspace/shared_workspace \
                    --device=/dev/infiniband/uverbs0 \
                    {0}/{1}:{2}
                    """.format(dockerhub_user, dockerhub_repo, dockerhub_tag)

ssh_client.run_on_all(launch_cont)
ssh_client.run_on_all("docker images > ~/imagelog")


################################################################
# Launch Training
# Run training in background thread so it will continue
# if disconnected from instance.
# To run not in background, remove `nohup` and `&> ~/shared_workspace/logs/out.log &`
################################################################
from datetime import datetime
import time
training_launch = """ 
mpirun --allow-run-as-root \
            -x FI_PROVIDER=\\\"efa\\\" \
            -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efa/lib:/usr/local/lib:/nccl/build/lib:/aws-ofi-nccl/install/lib \
            -x NCCL_DEBUG=INFO \
             -x NCCL_TREE_THRESHOLD=0 \
             --hostfile /root/.ssh/hosts \
             --mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 \
             --mca btl_vader_single_copy_mechanism none \
             --mca oob_tcp_if_include ens5 \
             --mca btl_tcp_if_include ens5 \
             python /workspace/shared_workspace/deep-learning-models/models/vision/detection/tools/train_docker.py \
             --configuration /workspace/shared_workspace/deep-learning-models/models/vision/detection/configs/docker_default_config.py \
             --base_learning_rate 15e-3 \
             --batch_size_per_device 4 \
             --fp16 True \
             --schedule 1x \
             --warmup_init_lr_scale 3.0 \
             --warmup_steps 1000 \
             --use_rcnn_bn False \
             --use_conv True \
             --ls 0.0 \
             --epochs 1 \
             --name demo

"""

ssh_client.run_on_master('mkdir -p ~/shared_workspace/logs')
ssh_client.run_on_all('date > time1')
print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
start = time.time()
training_thread = ssh_client.run_on_master("""docker exec mpicont bash -c \"{}\" &> ~/shared_workspace/logs/out.log&""".format(training_launch))
print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
end = time.time()'''
################################################################
# Cleanup and shutdown
# disconnect from notebook
# stop docker container
# terminate instance
# if you would rather just stop instances so they 
# can be used again later use
# ec2_client.stop_instances(InstanceIds=instances)
# ec2_client.start_instances(InstanceIds=instances)
################################################################

#notebook.disconnect()
#ssh_client.run_on_all("docker stop mpicont")
sleep(3000)
ssh_client.run_on_all("python ~/shared_workspace/logs/parse_and_submit.py ~/shared_workspace/logs/out.log 8 32 p3dn.24xlarge EC2 > parselog")
ec2_client.stop_instances(InstanceIds=instances)