from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from datetime import datetime
import os
import argparse
import importlib


def main(args):
    loader = importlib.machinery.SourceFileLoader('', args.configuration)
    cfg = loader.load_module()
    role = get_execution_role()
    main_script = 'tools/train.py'
    docker_image = cfg.sagemaker_user['docker_image']
    hvd_instance_count = cfg.sagemaker_user['hvd_instance_count']
    hvd_instance_type = cfg.sagemaker_user['hvd_instance_type']
    distributions = cfg.distributions
    output_path = cfg.sagemaker_job['output_path']
    job_name = cfg.sagemaker_job['job_name']
    channels = cfg.channels

    configuration = {
        'config': args.configuration,
        'amp': 'True',
        'autoscale-lr': 'True',
        'validate': 'True'
    }

    estimator = TensorFlow(
                    entry_point=main_script, 
                    source_dir='.',
                    image_name=docker_image, 
                    role=role,
                    framework_version="2.1.0",
                    py_version="py3",
                    train_instance_count=hvd_instance_count,
                    train_instance_type=hvd_instance_type,
                    distributions=distributions,
                    output_path=output_path,
                    train_volume_size=200,
                    hyperparameters=configuration)
    estimator.fit(channels, wait=False, job_name=job_name)
    print("Launched SageMaker job:", job_name)

def parse():
    """
    Parse path to configuration file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="SM Job configuration file")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse()
    main(args)
