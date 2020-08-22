# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Changelog (this file).
- Backbone models (ResNet variants, HRNEt, DarkNet53) under classification directory
- Backbone training modules with support for mixup augmentation
- V2 MomentumOptimizer for object detection framework
- Support for Sync Batchnorm
- Support for loading backbone models in SavedModel format
- Support to freeze model variables in object detection framework via configuration

### Changed
- Bugfix for ROIAlign layer
- Sampling of anchor targets changed to include low quality matches as well
- Weight initialization for FPN layer
- Use L1 loss instead of Smooth L1 for FRCNN family of models
- Cascade loss calculation update
- Mask RCNN segmentation crop/resize fix
- Switched to use crop and resize modification from TensorPack instead of tf.crop_and_resize
- Object detection configuration system changed from flat definitions to heirarchical structure where individual sections can be overwritten as per training requirement
- MPI and NCCL options updated for train scripts and SM job launcher
- Update README and posted top-1 accuracy for backbone training modules under classification directory
- Support for CI service via custom configurations

## [Unreleased]
### Added
- Changelog (this file).
- BERT model.
- Weights & Biases integration.
- ELECTRA model.
- Option in bbox target to return foreground assignments. Vector of indices of target within GT
- Ability for eval hooks to automatically detect masks in runner
- Mask target class that matches mask head output with GT masks
- Option for coco dataset to return masks
- Mask head and extractor options to faster RCNN
- Mask loss
- Mask head
- Profiler hook
- Mask rcnn configuration files
- RetinaNet model
- docs under vision/detection directory has README with results and setup instructions per model
- Generic AWSDet tutorial
- Ability to use Keras released backbone or custom SavedModel format backbone
- Ability to resume complete training state for object detection trainer to restart training from a saved trainer state

### Changed
- SageMaker and Kubernetes Dockerfiles have been merged into one.
- Use the module system rather than $PYTHONPATH, so jobs are launched with `python -m albert.run_pretraining` instead of `python albert/run_pretraining.py`.
- NLP models use `--per_gpu_batch_size` instead of `--batch_size`.
- NLP models use `--squad_version` instead of `--task_name`.
- NLP models use `--filesystem_prefix` instead of `--fsx_prefix`. This option is mostly hidden from the user and should be a no-op.
- NLP scripts start training at step 1 instead of step 0. So a job with `--total_steps=100` will run steps [1..100] instead of [0..99].
- NLP transformers dependency is now pinned to 2.11.0 instead of a custom fork. This removes the `--pre_layer_norm=true` option.
- Removed the `--pretrain_dataset` argument, now pass directly `--filesystem_prefix`, `--train_dir` and `--val_dir`.
- Background assignment in box target now uses while loop to handle rare case of too few backgrounds after initial duplicate assignment
- Switched COCO utils segmentation assignment to use yxyx instead of xyxy
- Matplotlib backend for visualization
- Directory structure has changed for vision models
- Per model configurations for EC2 and SageMaker have been introduced
- Now we have a single training entrypoint for both EC2 and SM training jobs
- Changes to core to support single stage detectors

### Removed
- NGC GPUMonitor Dockerfile.
- Duplicate code for schedulers, sagemaker trainers

## [0.2] - 2020-05-22
### Added
- ALBERT training scripts.
- Draft of computer vision framework.

### Changed
- ResNet training scripts moved to /legacy.

## [0.1] - 2020-05-01
### Added
- ResNet training scripts.
