from datetime import datetime
import os
import argparse
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
import pprint

def format_hyperparameters(args):
    hyperparameters = {
        'schedule': args.schedule,
        'fp16': (args.fp16 == 'True'),
        'base_learning_rate': float(args.base_learning_rate),
        'warmup_steps': int(args.warmup_steps),
        'warmup_init_lr_scale': float(args.warmup_init_lr_scale),
        'instance_type': args.instance_type,
        'instance_count': args.instance_count,
        'batch_size_per_device': args.batch_size_per_device,
        'num_workers_per_host': args.num_workers_per_host,
        'use_conv': (args.use_conv == 'True'),
        'use_rcnn_bn': (args.use_rcnn_bn == 'True'),
        'ls': args.ls
    }
    return hyperparameters
    
def main(args):
    hyperparameters = format_hyperparameters(args)
    now = datetime.now()
    distributions = {
    "mpi": {
        "enabled": True,
        "processes_per_host": args.num_workers_per_host,
        "custom_mpi_options": "-x OMPI_MCA_btl_vader_single_copy_mechanism=none -x TF_CUDNN_USE_AUTOTUNE=0"
        }
    }
    channels = {
        'coco': args.data_channel,
        'weights': args.weights_channel
    }
    s3_path = os.path.join(args.s3_path, time_str)
    job_name = '{}-{}-{}'.format(args.user_id, args.instance_name, time_str)
    output_path = os.path.join(s3_path, "output", job_name)
    configuration = {
        'configuration': 'configs/sagemaker_default_model_config.py', 
        's3_path': s3_path,
        'instance_name': job_name
    }
    configuration.update(hyperparameters)
    estimator = TensorFlow(
                entry_point=args.main_script, 
                source_dir=args.source_dir, 
                image_name=args.image, 
                role=args.sagemaker_role,
                framework_version="2.1.0",
                py_version="py3",
                train_instance_count=args.instance_count,
                train_instance_type=args.instance_type,
                distributions=distributions,
                output_path=output_path, train_volume_size=200,
                hyperparameters=configuration
    )
    
    estimator.fit(channels, wait=False, job_name=job_name)
    
    print("Started Sagemaker job: {}".format(job_name))
    pprint(configuration)
    
    

def parse():
    """
    Parse path to configuration file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", help="User ID for instance naming")
    parser.add_argument("--sagemaker_role", help="Sagemaker execution role")
    parser.add_argument("--image", help="Sagemaker Docker image")
    parser.add_argument("--source_dir", help="Training source dir")
    parser.add_argument("--instance_name", help="Sagemaker instance name")
    parser.add_argument("--s3_path", help="s3 path")
    parser.add_argument("--data_channel", help="s3 data path")
    parser.add_argument("--weights_channel", help="s3 weights path")
    parser.add_argument("--configuration", help="Model configuration file", default="configs/sagemaker_default_model_config.py")
    parser.add_argument("--main_script", help="entrypoint for training script", default="tools/train_sagemaker.py")
    parser.add_argument("--instance_count", help="Number of instances", default=1, type=int)
    parser.add_argument("--instance_type", help="Instance type for a worker", default="ml.p3dn.24xlarge")
    parser.add_argument("--num_workers_per_host", help="Number of workers on each instance", default=8, type=int)
    parser.add_argument("--model_dir", help="Location of model on Sagemaker instance")
    parser.add_argument("--base_learning_rate", help="float", default=15e-3, type=float)
    parser.add_argument("--batch_size_per_device", help="integer", default=2, type=int)
    parser.add_argument("--fp16", help="boolean", default="True")
    parser.add_argument("--schedule", help="learning rate schedule type", default="1x")
    parser.add_argument("--warmup_init_lr_scale", help="float")
    parser.add_argument("--warmup_steps", help="int", default=500, type=int)
    parser.add_argument("--epochs", help="int", default=13, type=int)
    parser.add_argument("--use_rcnn_bn", help="bool", default="False")
    parser.add_argument("--use_conv", help="bool", default="True")
    parser.add_argument("--ls", help="float", default=0.0, type=Float)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse()
    main(args)
    