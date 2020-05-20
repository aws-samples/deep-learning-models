import argparse
import os

from arguments import populate_pretraining_parser
from sagemaker_utils import launch_sagemaker_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # SageMaker parameters
    parser.add_argument(
        "--source_dir",
        help="For example, /Users/myusername/Desktop/deep-learning-models/models/nlp/albert",
    )
    parser.add_argument("--entry_point", default="run_pretraining.py")
    parser.add_argument("--role", default=os.environ["SAGEMAKER_ROLE"])
    parser.add_argument("--image_name", default=os.environ["SAGEMAKER_IMAGE_NAME"])
    parser.add_argument("--fsx_id", default=os.environ["SAGEMAKER_FSX_ID"])
    parser.add_argument(
        "--subnet_ids", help="Comma-separated string", default=os.environ["SAGEMAKER_SUBNET_IDS"]
    )
    parser.add_argument(
        "--security_group_ids",
        help="Comma-separated string",
        default=os.environ["SAGEMAKER_SECURITY_GROUP_IDS"],
    )

    # Instance specs
    parser.add_argument(
        "--instance_type",
        type=str,
        default="ml.p3dn.24xlarge",
        choices=["ml.p3dn.24xlarge", "ml.p3.16xlarge", "ml.g4dn.12xlarge"],
    )
    parser.add_argument("--instance_count", type=int, default=1)

    # Training script parameters
    populate_pretraining_parser(parser)
    args = parser.parse_args()

    args_dict = args.__dict__
    # Pop off the SageMaker parameters
    source_dir = args_dict.pop("source_dir")
    entry_point = args_dict.pop("entry_point")
    role = args_dict.pop("role")
    image_name = args_dict.pop("image_name")
    fsx_id = args_dict.pop("fsx_id")
    subnet_ids = args_dict.pop("subnet_ids").replace(" ", "").split(",")
    security_group_ids = args_dict.pop("security_group_ids").replace(" ", "").split(",")
    instance_type = args_dict.pop("instance_type")
    instance_count = args_dict.pop("instance_count")
    # Only the script parameters remain
    hyperparameters = {"fsx_prefix": "/opt/ml/input/data/training"}
    for key, value in args_dict.items():
        if value is not None:
            hyperparameters[key] = value

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
