# ALBERT (A Lite BERT)

TensorFlow 2.1 implementation of pretraining and finetuning scripts for ALBERT, a state-of-the-art language model.

The original paper: [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

### Overview

Language models help AWS customers to improve search results, text classification, question answering, and customer service routing. BERT and its successive improvements are incredibly powerful, yet complex to pretrain. Here we demonstrate how to train a faster, smaller, more accurate BERT-based model called [ALBERT](https://arxiv.org/abs/1909.11942) on Amazon SageMaker with the FSx filesystem and TensorFlow 2 models from [huggingface/transformers](https://github.com/huggingface/transformers). We can pretrain ALBERT for a fraction of the cost of BERT and achieve better accuracy on SQuAD.

![SageMaker -> EC2 -> FSx infrastructure diagram](https://user-images.githubusercontent.com/4564897/81020280-b207a100-8e25-11ea-8b57-38f0a09a7fb2.png
)


### Results
All training is done with sequence length 512.

| Model | Nodes | Global Batch Size | Batch size per GPU | Gradient accumulation steps | iterations/sec | Steps | Time-to-train | SQuADv2 F1/EM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| albert-base | 1 | 4096 | 32 | 16 | 0.24 | 125000 | 144 hours | same |
| albert-base | 2 | 4096 | 32 | 8 | 0.46 | 125000 | 75 hours | same |
| albert-base | 4 | 4096 | 32 | 4 | 0.90 | 125000 | 38 hours | same |
| albert-base | 8 | 4096 | 32 | 2 | 1.73 | 125000 | 20 hours | 78.4/75.2 |


### How To Launch Training

All commands should be run from the `models/nlp` directory.

1. Create an FSx volume.

2. Download the datasets onto FSx. The simplest way to start is with English Wikipedia. The structure should be as follows:

```
/fsx
    /deep-learning-models
    /logs
        /albert
    /tensorboard
    /checkpoints
        /albert
    /albert_data
        /train
        /val
```

3. Create an Amazon Elastic Container Registry (ECR) repository. Then build a Docker image from `models/nlp/Dockerfile` and push it to ECR.

```bash
export ACCOUNT_ID=
export REPO=
export IMAGE=${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${REPO}:py37_tf211
docker build -t ${IMAGE} .
# AWS-CLI v1
$(aws ecr get-login --no-include-email)
# AWS-CLI v2
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com
docker push ${IMAGE}
```

4. Define environment variables to point to the FSx volume. For a list, use a comma-separated string.

```bash
export SAGEMAKER_ROLE=arn:aws:iam::${ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20200101T123
export SAGEMAKER_IMAGE_NAME=${IMAGE}
export SAGEMAKER_FSX_ID=fs-123
export SAGEMAKER_FSX_MOUNT_NAME=fsx
export SAGEMAKER_SUBNET_IDS=subnet-123
export SAGEMAKER_SECURITY_GROUP_IDS=sg-123,sg-456
```

5. Define environment variables for directories. If your data was in /fsx/albert_data/train, you would use:

```bash
export TRAIN_DIR=albert_data/train
export VAL_DIR=albert_data/validation
export LOG_DIR=logs/albert
export CHECKPOINT_DIR=checkpoints/electra
```

6. Launch the SageMaker job.

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_pretraining.py \
    --sm_job_name=albert-pretrain \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=1 \
    --train_dir=${TRAIN_DIR} \
    --val_dir=${VAL_DIR} \
    --log_dir=${LOG_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --load_from=scratch \
    --model_type=albert \
    --model_size=base \
    --per_gpu_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --warmup_steps=3125 \
    --total_steps=125000 \
    --learning_rate=0.00176 \
    --optimizer=lamb \
    --log_frequency=10 \
    --name=myfirstjob
```

7. Launch a SageMaker finetuning job.

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_squad.py \
    --sm_job_name=albert-squad \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=1 \
    --train_dir=${TRAIN_DIR} \
    --val_dir=${VAL_DIR} \
    --log_dir=${LOG_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --load_from=scratch \
    --model_type=albert \
    --model_size=base \
    --per_gpu_batch_size=6 \
    --total_steps=8144 \
    --warmup_steps=814 \
    --learning_rate=3e-5 \
    --squad_version=squadv2
```

8. Enter the Docker container to debug and edit code.

```bash
docker run -it --privileged -v=/fsx:/fsx --gpus=all --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 --rm ${IMAGE} /bin/bash
```

<!-- ### Training results. These will be posted shortly. -->

### Command-Line Parameters

See `common/arguments.py` for a complete list. Here are the main ones:

Loading from checkpoint:
- `model_type`: One of "albert", "bert", "electra".
- `model_size`: One of "small", "base", "large".
- `load_from`: One of "scratch", "checkpoint", "huggingface". If checkpoint, then checkpoint_path is required.
- `checkpoint_path`: For example: "/fsx/checkpoints/albert/2020...step125000". No .ckpt on the end.
- `load_optimizer_state`: One of "true", "false".
