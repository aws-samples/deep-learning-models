## Using Weights and Biases
```bash
pip install wandb
wandb login ${WANDB_API_KEY}
```

## Hyperparameters
ALBERT used LAMB, 0.00176 learning rate with linear warmup and decay, global batch size 4096.
Data was seq_len 512 with max 20 masks.

BERT uses batch size 256. epsilon=1e-6.

RoBERTa has better results with bsz 2K for 125k steps.
RoBERTa uses Adam. They change B_2 to 0.98. Also epsilon=1e-6, sensitive.

```bash
python run_pretraining.py \
    --load_from=scratch \
    --model_type=bert \
    --model_size=base \
    --per_gpu_batch_size=16 \
    --warmup_steps=10000 \
    --phase1_steps=900000 \
    --phase2_steps=100000 \
    # --total_steps=1000000 \
    --optimizer=adamw \
    --learning_rate=1e-5 \
    --weight_decay=0.01 \
    --adam_beta_2=0.98 \
    --adam_epsilon=1e-6 \
    --hidden_dropout_prob=0.1 \
    --attention_probs_dropout_prob=0.1 \
    --phase_1_
```

For finetuning:
```bash
python run_finetuning.py \
    --global_batch_size=48 \
    --learning_rate=1.5e-5 \
```
The BERT training loss is the sum of MLM likelihood and NSP likelihood. Do they even use cross-entropy?

PruneBERT uses seq_len 128 all the way through. Learning rate 3e-5.
