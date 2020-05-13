#!/bin/bash

docker stop tensorflow
docker run --gpus all -it --rm -d --net=host --name tensorflow -v ~/shared_workspace/:/workspace/shared_workspace johnbensnyder/ngc:20.02 bash
docker exec tensorflow /bin/sh -c 'cd shared_workspace/mmdetection_tf; mpirun -np 8 --host localhost:8 --allow-run-as-root python /workspace/shared_workspace/mmdetection_tf/train_hvd.py' > log.txt
python parse_and_submit.py log.txt  8 2 p3.16xlarge EC2
