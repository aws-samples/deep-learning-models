# BERT

TensorFlow 2.1 implementation of pretraining and finetuning scripts for BERT.

The original paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

### How To Launch Training

All commands should be run from the `models/nlp` directory.

1. Create an FSx volume.

2. Download the datasets onto FSx. The simplest way to start is with English Wikipedia.

3. Create an Amazon Elastic Container Registry (ECR) repository. Then build a Docker image from `models/nlp/Dockerfile` and push it to ECR.

```bash
export ACCOUNT_ID=
export REPO=
export IMAGE=${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${REPO}:py37_tf211
docker build -t ${IMAGE} .
$(aws ecr get-login --no-include-email)
docker push ${IMAGE}
```

4. Define environment variables to point to the FSx volume. For a list, use a comma-separated string.

```bash
export SAGEMAKER_ROLE=arn:aws:iam::${ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20200101T123
export SAGEMAKER_IMAGE_NAME=${IMAGE}
export SAGEMAKER_FSX_ID=fs-123
export SAGEMAKER_SUBNET_IDS=subnet-123
export SAGEMAKER_SECURITY_GROUP_IDS=sg-123,sg-456
```

5. Define BERT-specific run names.

```bash
export PHASE1_RUN_NAME=bertphase1
export PHASE2_RUN_NAME=bertphase2
export PHASE1_STEPS=14076
```

6. Launch the SageMaker Phase1 training.

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_pretraining.py \
    --sm_job_name=bert-pretrain-phase1 \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=8 \
    --load_from=scratch \
    --model_type=bert \
    --model_size=base \
    --per_gpu_batch_size=128 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --optimizer=lamb \
    --gradient_accumulation_steps=4 \
    --run_name=${PHASE1_RUN_NAME} \
    --warmup_steps=1400 \
    --total_steps=${PHASE1_STEPS} \
    --learning_rate=0.00176 \
    --squad_freq=0 \
    --log_frequency=10 \
    --name=mybertphase1
```

7. Launch the SageMaker Phase2 training.

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_pretraining.py \
    --sm_job_name=bert-pretrain-phase1 \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=8 \
    --load_from=checkpoint \
    --model_type=bert \
    --model_size=base \
    --per_gpu_batch_size=32 \
    --max_seq_length=512 \
    --max_predictions_per_seq=80 \
    --optimizer=lamb \
    --gradient_accumulation_steps=8 \
    --run_name=${PHASE2_RUN_NAME} \
    --warmup_steps=312 \
    --total_steps=3124 \
    --learning_rate=0.00176 \
    --log_frequency=10 \
    --checkpoint_path=/fsx/checkpoints/bert/${RUN_NAME_PHASE1}-step${PHASE1_STEPS} \
    --name=mybertphase2
```

8. Launch a SageMaker finetuning job.

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_squad.py \
    --sm_job_name=bert-squad \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=1 \
    --load_from=scratch \
    --model_type=bert \
    --model_size=base \
    --per_gpu_batch_size=6 \
    --total_steps=3649 \
    --warmup_steps=365 \
    --learning_rate=5e-5 \
    --task_name=squadv1
```

9. Enter the Docker container to debug and edit code.

```bash
docker run -it -v=/fsx:/fsx --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm ${IMAGE} /bin/bash
```
