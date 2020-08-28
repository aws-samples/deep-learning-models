import os.path as osp

# date time settings to update paths for jobs
from datetime import datetime
now = datetime.now()
time_str = now.strftime("%d-%m-%Y-%H-%M")
date_str = now.strftime("%d-%m-%Y")


# sagemaker settings
sagemaker_user=dict(
    user_id='CI',
    s3_bucket='mzanur-sagemaker',
    docker_image='578276202366.dkr.ecr.us-east-1.amazonaws.com/mzanur-awsdet-ecr:awsdet',
    hvd_processes_per_host=8,
    hvd_instance_type='ml.p3.16xlarge', #'ml.p3dn.24xlarge',
    hvd_instance_count=2,
)
# settings for distributed training on sagemaker
mpi_options="\
-x OMPI_MCA_plm_rsh_no_tree_spawn=1 -bind-to none -map-by slot -x OMPI_MCA_pml=ob1 \
-x OMPI_MCA_btl_vader_single_copy_mechanism=none \
-x OMPI_MCA_btl=tcp,self \
-x NCCL_TREE_THRESHOLD=4294967296 \
-x HOROVOD_CYCLE_TIME=0.5 \
-x HOROVOD_FUSION_THRESHOLD=67108864"

distributions=dict(
    mpi=dict(
        enabled=True,
        processes_per_host=sagemaker_user['hvd_processes_per_host'],
        custom_mpi_options=mpi_options,
    )
)

# sagemaker channels
channels=dict( 
    coco='s3://{}/awsdet/data/coco/'.format(sagemaker_user['s3_bucket']),
    weights='s3://{}/awsdet/data/weights/'.format(sagemaker_user['s3_bucket'])
)

job_str='{}x{}-{}'.format(sagemaker_user['hvd_instance_count'], sagemaker_user['hvd_processes_per_host'], time_str)
sagemaker_job=dict(
    s3_path='s3://{}/retinanet/outputs/{}'.format(sagemaker_user['s3_bucket'], time_str),
    job_name='{}-retinanet-{}'.format(sagemaker_user['user_id'], job_str),
    output_path='',
)
sagemaker_job['output_path']='{}/output/{}'.format(sagemaker_job['s3_path'], sagemaker_job['job_name'])
