# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Changelog (this file).
- BERT model.
- Weights & Biases integration.

### Changed
- SageMaker and Kubernetes Dockerfiles have been merged into one.
- Use the module system rather than $PYTHONPATH, so jobs are launched with `python -m albert.run_pretraining` instead of `python albert/run_pretraining.py`.

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
