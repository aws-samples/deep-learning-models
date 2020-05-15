import argparse
import os

from sagemaker_utils import launch_sagemaker_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # SageMaker parameters
    parser.add_argument(
        "--source_dir",
        help="For example, /Users/myusername/Desktop/deep-learning-models/models/nlp/albert",
    )
    parser.add_argument("--entry_point", type=str, default="run_squad.py")
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
        choices=["ml.p3dn.24xlarge", "ml.p3.16xlarge"],
    )
    parser.add_argument("--instance_count", type=int, default=1)
    # Training script parameters
    # None are required because defaults are in run_squad.py
    parser.add_argument("--load_from", required=True)
    parser.add_argument("--model_type")
    parser.add_argument("--model_size")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--data_dir")
    parser.add_argument("--dataset")
    parser.add_argument("--name")
    parser.add_argument("--validate_frequency", type=int)
    parser.add_argument("--checkpoint_frequency", type=int)
    parser.add_argument("--skip_xla", choices=["true"])
    parser.add_argument("--eager", choices=["true"])
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
    job_name = f"squad-{instance_count}x{instance_abbr}-{args.load_from}"

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
