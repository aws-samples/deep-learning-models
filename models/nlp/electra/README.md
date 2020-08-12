
# ELECTRA

TensorFlow 2.1 implementation of pretraining and finetuning scripts for ELECTRA.

The original paper: [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

* We pretrain 125000 steps with a total batch size of 1024 for maximum sequence length of 512 across 8 p3dn.24xlarge nodes.
* We finetune SQuAD v2.0 5430 steps with a total batch size of 48 on a single p3dn.24xlarge node.

SQuAD F1 score combines both precision and recall of each word in the predicted answer ranging between 0-100.

| Model | Total Training Time | SQuAD v2.0 EM | SQuAD v2.0 F1 |
| --- | --- | --- | --- |
| ELECTRA-small | 11 hrs 40 min | 68.27 | 71.56 |

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
export SAGEMAKER_SUBNET_IDS=subnet-123
export SAGEMAKER_SECURITY_GROUP_IDS=sg-123,sg-456
```

5. Define ELECTRA-specific run names.

```bash
export RUN_NAME=myelectrapretraining
export TOTAL_STEPS=125000
# The data should be in TFRecords inside $TRAIN_DIR on the FSx volume
export TRAIN_DIR=electra_data/train
export VAL_DIR=electra_data/val
export LOG_DIR=logs/electra
export CHECKPOINT_DIR=checkpoints/electra
```
6. Launch the SageMaker Electra pretraining.

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=electra/run_pretraining.py \
    --sm_job_name=electra-pretrain \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=8 \
    --train_dir=${TRAIN_DIR} \
    --val_dir=${VAL_DIR} \
    --log_dir=${LOG_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --load_from=scratch \
    --model_type=electra \
    --model_size=small \
    --attention_probs_dropout_prob=0.1 \
    --hidden_dropout_prob=0.1 \
    --checkpoint_frequency=10000 \
    --per_gpu_batch_size=16 \
    --max_seq_len=512 \
    --learning_rate=2e-3 \
    --end_learning_rate=4e-4 \
    --weight_decay=0.01 \
    --warmup_steps=10000 \
    --validation_frequency=10000 \
    --total_steps=${TOTAL_STEPS} \
    --log_frequency=2000 \
    --run_name=${RUN_NAME} \
    --name=myelectra
```

7. Launch a SageMaker finetuning job.


SQuAD v2.0

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_squad.py \
    --sm_job_name=electra-squadv2 \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=1 \
    --load_from=checkpoint \
    --checkpoint_path=checkpoints/electra/${RUN_NAME}-step${TOTAL_STEPS}-discriminator \
    --model_type=electra \
    --model_size=small \
    --per_gpu_batch_size=6 \
    --weight_decay=0 \
    --model_size=small \
    --squad_version=squadv2 \
    --learning_rate=40e-5 \
    --warmup_steps=543 \
    --total_steps=5430 \
    --validation_frequency=50000 \
    --evaluate_frequency=50000 \
    --skip_xla=true
```

8. Enter the Docker container to debug and edit code.

```bash
docker run -it --privileged -v=/fsx:/fsx --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm ${IMAGE} /bin/bash
```

### Command-Line Parameters
See [common/arguments.py](common/arguments.py) for full details.
