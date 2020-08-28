# BERT

TensorFlow 2.1 implementation of pretraining and finetuning scripts for BERT.

The original paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

Pretraining consists of two phases. We use mixed-batch training.

* Phase1: We pretrain 14064 steps with a total batch size of 32K for maximum sequence length of 128 across 8 p3dn.24xlarge nodes.
* Phase2: We pretrain 6248 steps with a total batch size of 8K for maximum sequence length of 512 across 8 p3dn.24xlarge nodes.
* Lastly, we finetune SQuAD v1.1 3649 steps with a total batch size of 48 on a single p3dn.24xlarge node.

SQuAD F1 score combines both precision and recall of each word in the predicted answer ranging between 0-100. For fewer nodes, we apply gradient accumulation to reach the same global batch size per step.

| Model | p3dn Nodes | Phase1 | Phase2 | Finetuning | Total Training Time | SQuAD v1.1 F1 | SQuAD v2.0 F1 |
| --- | --- | --- | --- |  --- | --- | --- | --- |
| BERT-base | 1 | 32 hours | 15 hours | 15 mins | 47 hours | same | same |
| BERT-base | 2 | 17 hours | 8 hours | 15 mins | 25 hours | same | same |
| BERT-base | 4 | 10 hours | 4 hours | 15 mins | 14 hours | same | same |
| BERT-base | 8 | 5 hrs 33 mins | 2 hrs 53 mins | 15 mins | 8 hrs 41 mins | 87.68 | 76.14 |


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
export PHASE1_STEPS=14064
export PHASE2_STEPS=6248
# The data should be in TFRecords inside /fsx/${TRAIN_DIR}
export TRAIN_DIR=bert_data/train
export VAL_DIR=bert_data/val
export LOG_DIR=logs/bert
export CHECKPOINT_DIR=checkpoints/bert
```

6. Launch the SageMaker Phase1 training.

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_pretraining.py \
    --sm_job_name=bert-pretrain-phase1 \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=8 \
    --train_dir=${TRAIN_DIR} \
    --val_dir=${VAL_DIR} \
    --log_dir=${LOG_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --load_from=scratch \
    --model_type=bert \
    --model_size=base \
    --per_gpu_batch_size=128 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --optimizer=lamb \
    --learning_rate=0.005 \
    --end_learning_rate=0.0003 \
    --hidden_dropout_prob=0.1 \
    --attention_probs_dropout_prob=0.1 \
    --gradient_accumulation_steps=4 \
    --learning_rate_decay_power=0.5 \
    --warmup_steps=2812 \
    --total_steps=${PHASE1_STEPS} \
    --log_frequency=100 \
    --squad_frequency=0 \
    --run_name=${PHASE1_RUN_NAME} \
    --name=mybertphase1
```

7. Launch the SageMaker Phase2 training.

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_pretraining.py \
    --sm_job_name=bert-pretrain-phase2 \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=8 \
    --train_dir=${TRAIN_DIR} \
    --val_dir=${VAL_DIR} \
    --log_dir=${LOG_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --load_from=checkpoint \
    --load_optimizer_state=false \
    --model_type=bert \
    --model_size=base \
    --per_gpu_batch_size=32 \
    --max_seq_length=512 \
    --max_predictions_per_seq=80 \
    --optimizer=lamb \
    --learning_rate=0.004 \
    --end_learning_rate=0.0003 \
    --hidden_dropout_prob=0.1 \
    --attention_probs_dropout_prob=0.1 \
    --gradient_accumulation_steps=4 \
    --learning_rate_decay_power=0.5 \
    --warmup_steps=625 \
    --total_steps=${PHASE2_STEPS} \
    --log_frequency=100 \
    --squad_frequency=0 \
    --run_name=${PHASE2_RUN_NAME} \
    --checkpoint_path=checkpoints/albert/${PHASE1_RUN_NAME}-step${PHASE1_STEPS} \
    --name=mybertphase2
```

8. Launch a SageMaker finetuning job.

For SQuAD v1.1

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_squad.py \
    --sm_job_name=bert-squadv1 \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=1 \
    --load_from=checkpoint \
    --checkpoint_path=checkpoints/albert/${PHASE2_RUN_NAME}-step${PHASE2_STEPS} \
    --model_type=bert \
    --per_gpu_batch_size=6 \
    --model_size=base \
    --squad_version=squadv1 \
    --learning_rate=5e-5 \
    --warmup_steps=365 \
    --total_steps=3649 \
    --validation_frequency=10000 \
    --evaluate_frequency=10000 \
    --skip_xla=true \
```

SQuAD v2.0

```bash
python -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_squad.py \
    --sm_job_name=bert-squadv2 \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=1 \
    --load_from=checkpoint \
    --checkpoint_path=checkpoints/albert/${PHASE2_RUN_NAME}-step${PHASE2_STEPS} \
    --model_type=bert \
    --per_gpu_batch_size=6 \
    --model_size=base \
    --squad_version=squadv2 \
    --learning_rate=10.0e-5 \
    --warmup_steps=814 \
    --total_steps=8144 \
    --validation_frequency=10000 \
    --evaluate_frequency=100000 \
    --skip_xla=true \
```

9. Enter the Docker container to debug and edit code.

```bash
docker run -it --privileged -v=/fsx:/fsx --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm ${IMAGE} /bin/bash
```
