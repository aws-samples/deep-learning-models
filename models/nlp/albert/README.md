# ALBERT (A Lite BERT)

TensorFlow 2.1 implementation of pretraining and finetuning scripts for ALBERT, a state-of-the-art language model.

The original paper: [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

### Overview

Language models help AWS customers to improve search results, text classification, question answering, and customer service routing. BERT and its successive improvements are incredibly powerful, yet complex to pretrain. Here we demonstrate how to train a faster, smaller, more accurate BERT-based model called [ALBERT](https://arxiv.org/abs/1909.11942) on Amazon SageMaker with the FSx filesystem and TensorFlow 2 models from [huggingface/transformers](https://github.com/huggingface/transformers). We can pretrain ALBERT for a fraction of the cost of BERT and achieve better accuracy on SQuAD.

![SageMaker -> EC2 -> FSx infrastructure diagram](https://user-images.githubusercontent.com/4564897/81020280-b207a100-8e25-11ea-8b57-38f0a09a7fb2.png
)

### How To Launch Training

1. Create an FSx volume.

2. Download the datasets onto FSx. You will need English Wikipedia and BookCorpus, and helper scripts for downloading will be forthcoming.

3. Create an Elastic Container Registry repository. Then build a Docker image from `docker/ngc_sagemaker.Dockerfile` and push it to ECR.

```bash
export IMAGE=${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${REPO}:ngc_sagemaker
docker build -t ${IMAGE} -f docker/ngc_sagemaker.Dockerfile .
$(aws ecr get-login --no-include-email)
docker push ${IMAGE}
```

3. Define variables to point to the FSx volume by modifying `fsx_settings.py`.

4. Launch the SageMaker job.

```bash
python sagemaker_pretraining.py \
    --source_dir=. \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=8 \
    --load_from=scratch \
    --model_type=albert \
    --model_size=base \
    --batch_size=32 \
    --gradient_accumulation_steps=2 \
    --warmup_steps=3125 \
    --total_steps=125000 \
    --learning_rate=0.00176 \
    --optimizer=lamb \
    --name=myfirstjob
```

<!-- ### Training results

These will be posted shortly. -->
