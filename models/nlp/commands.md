The EKS command-line arguments are a strict subset of the SageMaker arguments. SageMaker adds the following options:
--source_dir=.
--entry_point=albert/run_pretraining.py
--sm_job_name=bert-pretrain-phase1
--instance_type=ml.p3dn.24xlarge
--instance_count=8

# EKS
python -m albert.run_pretraining \
    --train_dir=albert_pretraining/tfrecords/train/max_seq_len_512_max_predictions_per_seq_20_masked_lm_prob_15 \
    --val_dir=albert_pretraining/tfrecords/validation/max_seq_len_512_max_predictions_per_seq_20_masked_lm_prob_15 \
    --log_dir=logs/albert \
    --checkpoint_dir=checkpoints/albert \
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
