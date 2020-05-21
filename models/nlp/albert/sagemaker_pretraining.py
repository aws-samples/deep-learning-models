import argparse

from common.arguments import populate_pretraining_parser, populate_sagemaker_parser
from common.sagemaker_utils import launch_sagemaker_job, pop_sagemaker_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    populate_sagemaker_parser(parser)
    populate_pretraining_parser(parser)
    args = parser.parse_args()

    args_dict = args.__dict__
    # Pop off the SageMaker parameters
    (
        source_dir,
        entry_point,
        role,
        image_name,
        fsx_id,
        subnet_ids,
        security_group_ids,
        instance_type,
        instance_count,
    ) = pop_sagemaker_args(args_dict)
    # Only the script parameters remain
    hyperparameters = dict()
    for key, value in args_dict.items():
        if value is not None:
            hyperparameters[key] = value
    hyperparameters["fsx_prefix"] = "/opt/ml/input/data/training"

    instance_abbr = {
        "ml.p3dn.24xlarge": "p3dn",
        "ml.p3.16xlarge": "p316",
        "ml.g4dn.12xlarge": "g4dn",
    }[instance_type]
    job_name = f"albert-pretrain-{instance_count}x{instance_abbr}"

    launch_sagemaker_job(
        job_name=job_name,
        source_dir=source_dir,
        entry_point=entry_point,
        instance_type=instance_type,
        instance_count=instance_count,
        hyperparameters=hyperparameters,
        role=role,
        image_name=image_name,
        fsx_id=fsx_id,
        subnet_ids=subnet_ids,
        security_group_ids=security_group_ids,
    )
