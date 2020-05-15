"""
SageMaker has multiple ways to specify a script entrypoint.
- "Upload" mode, where a local filepath is passed as a required Python function parameter.
- "Environment variable" mode, where $SAGEMAKER_PROGRAM, if set in the Docker image, will override the uploaded file.
To use a script located on FSx, use environment variable mode. It is necessary to pass a dummy script
to the Python function, but this will be ignored if the environment variable is set.

SageMaker will mount an input FSx channel not at /fsx, but at /opt/ml/input/data/training.
So data at /fsx/myfolder/ is actually at /opt/ml/input/data/training/myfolder/

When using a custom container, enabling SSH is necessary, as shown here:
https://github.com/aws/sagemaker-tensorflow-container/blob/master/docker/1.15.2/py3/Dockerfile.cpu#L45-L77
"""


from typing import Any, Dict, List

from sagemaker.inputs import FileSystemInput
from sagemaker.tensorflow import TensorFlow


def launch_sagemaker_job(
    job_name: str,
    source_dir: str,
    entry_point: str,
    instance_type: str,
    instance_count: int,
    hyperparameters: Dict[str, Any],
    role: str,
    image_name: str,
    fsx_id: str,
    subnet_ids: List[str],
    security_group_ids: List[str],
) -> None:
    """ Create a SageMaker job connected to FSx and Horovod. """
    hvd_processes_per_host = {"ml.p3dn.24xlarge": 8, "ml.p3.16xlarge": 8, "ml.g4dn.12xlarge": 4,}[
        instance_type
    ]
    distributions = {
        "mpi": {
            "enabled": True,
            "processes_per_host": hvd_processes_per_host,
            "custom_mpi_options": "-verbose --NCCL_DEBUG=INFO -x OMPI_MCA_btl_vader_single_copy_mechanism=none",
        }
    }
    # Create FSx input
    fsx_input = FileSystemInput(
        file_system_id=fsx_id,
        file_system_type="FSxLustre",
        directory_path="/fsx",
        file_system_access_mode="rw",
    )
    # Create the job template
    estimator_hvd = TensorFlow(
        base_job_name=job_name,
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        framework_version="2.1.0",
        py_version="py3",
        hyperparameters=hyperparameters,
        train_instance_count=instance_count,
        train_instance_type=instance_type,
        distributions=distributions,
        image_name=image_name,
        subnets=subnet_ids,
        security_group_ids=security_group_ids,
        enable_sagemaker_metrics=True,
    )
    # Launch the job
    estimator_hvd.fit(fsx_input)
