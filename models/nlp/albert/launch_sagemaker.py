import argparse
import dataclasses

from transformers import HfArgumentParser

from common.arguments import (
    DataTrainingArguments,
    LoggingArguments,
    ModelArguments,
    PathArguments,
    SageMakerArguments,
    TrainingArguments,
)
from common.sagemaker_utils import launch_sagemaker_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            LoggingArguments,
            PathArguments,
            SageMakerArguments,
        )
    )
    (
        model_args,
        data_args,
        train_args,
        log_args,
        path_args,
        sm_args,
    ) = parser.parse_args_into_dataclasses()

    hyperparameters = dict()
    for args in [model_args, data_args, train_args, path_args, log_args]:
        for key, value in dataclasses.asdict(args).items():
            if value is not None:
                hyperparameters[key] = value
    hyperparameters["filesystem_prefix"] = "/opt/ml/input/data/training"

    instance_abbr = {
        "ml.p3dn.24xlarge": "p3dn",
        "ml.p3.16xlarge": "p316",
        "ml.g4dn.12xlarge": "g4dn",
    }[sm_args.instance_type]
    job_name = f"{sm_args.sm_job_name}-{sm_args.instance_count}x{instance_abbr}"

    launch_sagemaker_job(
        hyperparameters=hyperparameters,
        job_name=job_name,
        source_dir=sm_args.source_dir,
        entry_point=sm_args.entry_point,
        instance_type=sm_args.instance_type,
        instance_count=sm_args.instance_count,
        role=sm_args.role,
        image_name=sm_args.image_name,
        fsx_id=sm_args.fsx_id,
        fsx_mount_name=sm_args.fsx_mount_name,
        subnet_ids=sm_args.subnet_ids,
        security_group_ids=sm_args.security_group_ids,
    )
