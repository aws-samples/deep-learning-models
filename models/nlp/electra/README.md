# Work in Progress

Check back later!

```python
horovodrun -H localhost:8 -np 8 \
python -m electra.run_pretraining \
    --per_gpu_batch_size=32 \
    --max_seq_len=128 \
    --learning_rate=5e-4 \
    --weight_decay=0.01 \
    --warmup_steps=10000 \
    --total_steps=1000000 \
    --log_frequency=20
```

```python
CUDA_VISIBLE_DEVICES=1,2 \
horovodrun -H localhost:2 -np 2 \
python -m electra.run_pretraining \
    --per_gpu_batch_size=128 \
    --max_seq_len=128 \
    --learning_rate=5e-4 \
    --weight_decay=0.01 \
    --warmup_steps=10000 \
    --total_steps=1000000 \
    --log_frequency=20
```
