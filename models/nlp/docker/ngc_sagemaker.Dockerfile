# Contains TensorFlow 2.1 and Horovod 0.19.0
FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

RUN apt-get update && \
    apt-get install -y iputils-ping net-tools openmpi-bin clustershell htop

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir scikit-learn gputil requests

ENV HDF5_USE_FILE_LOCKING "FALSE"

# Install SSH on SageMaker machines
RUN apt-get install -y --no-install-recommends openssh-client openssh-server
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN mkdir -p /root/.ssh/ && \
    mkdir -p /var/run/sshd && \
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config


RUN pip install --no-cache-dir \
    mpi4py \
    sagemaker-containers \
    wandb \
    tensorflow-addons==0.9.1
# TODO: Why does installing torch break TF XLA support?

RUN pip install git+git://github.com/jarednielsen/transformers.git@tfsquad
ENV PYTHONPATH "${PYTHONPATH}:/fsx/deep-learning-models/models/nlp"
ENV PYTHONPATH "${PYTHONPATH}:/opt/ml/input/data/training/deep-learning-models/models/nlp"
