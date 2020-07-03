# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- RetinaNet model
- docs under vision/detection directory has README with results and setup instructions per model
- Generic AWSDet tutorial
- Ability to use Keras released backbone or custom SavedModel format backbone
- Ability to resume complete training state for object detection trainer to restart training from a saved trainer state

### Changed
- Directory structure has changed for vision models
- Per model configurations for EC2 and SageMaker have been introduced
- Now we have a single training entrypoint for both EC2 and SM training jobs
- Changes to core to support single stage detectors

### Removed
- Duplicate code for schedulers, sagemaker trainers


## [Unreleased]
### Added
- Changelog (this file).
- BERT model.
- Weights & Biases integration.
- Draft of ELECTRA model.

### Changed
- SageMaker and Kubernetes Dockerfiles have been merged into one.
- Use the module system rather than $PYTHONPATH, so jobs are launched with `python -m albert.run_pretraining` instead of `python albert/run_pretraining.py`.
- NLP models use `--per_gpu_batch_size` instead of `--batch_size`.
- NLP scripts start training at step 1 instead of step 0. So a job with `--total_steps=100` will run steps [1..100] instead of [0..99].
- NLP transformers dependency is now pinned to 2.11.0 instead of a custom fork. This removes the `--pre_layer_norm=true` option.

### Removed
- NGC GPUMonitor Dockerfile.

## [0.2] - 2020-05-22
### Added
- ALBERT training scripts.
- Draft of computer vision framework.

### Changed
- ResNet training scripts moved to /legacy.

## [0.1] - 2020-05-01
### Added
- ResNet training scripts.
