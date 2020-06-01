## Using Weights and Biases
```bash
pip install wandb
wandb login ${WANDB_API_KEY}
```

## Hyperparameters
ALBERT used LAMB, 0.00176 learning rate with linear warmup and decay, global batch size 4096.
Data was seq_len 512 with max 20 masks.

BERT hyperparameters:
```bash
python run_pretraining.py \
    --model_type=bert \
    --warmup_steps=10000 \
    --phase1_steps=900000 \
    --phase2_steps=100000 \
    --optimizer=adam \
    --learning_rate=1e-4 \
    --beta_1=0.9 \
    --beta_2=0.999 \
    --weight_decay=0.01 \
    --hidden_dropout_prob=0.1 \
    --attention_probs_dropout_prob=0.1 \
    --global_batch_size=256
```

RoBERTa hyperparameters:
```bash
python run_pretraining.py \
    --model_type=bert \
    --total_steps=500000 \
    --warmup_steps=24000 \
    --optimizer=adamw \
    --learning_rate=6e-4 \
    --beta_1=0.9 \
    --beta_2=0.98 \
    --epsilon=1e-6 \
    --weight_decay=0.01 \
    --hidden_dropout_prob=0.1 \
    --attention_probs_dropout_prob=0.1 \
    --global_batch_size=8192
```

LAMB hyperparameters:
```bash
python run_pretraining.py \
    --model_type=bert \
    --optimizer=lamb \
    --learning_rate=0.01 \
    --beta_1=0.9 \
    --beta_2=0.999 \
    --global_batch_size=2048
```
