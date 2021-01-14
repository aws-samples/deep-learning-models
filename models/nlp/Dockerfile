FROM nvidia/cuda:11.0-devel-ubuntu18.04
# TF 2.4 works with CUDA 11.0, not 11.1 - https://github.com/tensorflow/tensorflow/issues/45848

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV HOROVOD_VERSION=0.21.1
ENV TENSORFLOW_PIP=tensorflow
ENV TENSORFLOW_VERSION=2.4.0
ENV TENSORFLOW_ADDONS_VERSION=0.12.0
ENV PYTORCH_VERSION=1.7.1
ENV TORCHVISION_VERSION=0.8.2
# cuDNN version listed here: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install
ENV CUDNN_VERSION=8.0.5.39-1+cuda11.0
ENV NCCL_VERSION=2.8.3-1+cuda11.0

ARG python=3.7
ENV PYTHON_VERSION=${python}

# LD_LIBRARY_PATH is set incorrectly for legacy compatibility; see https://gitlab.com/nvidia/container-images/cuda/-/issues/47
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs

# Solution to "Couldn't open CUDA library libcuda.so" at https://github.com/tensorflow/tensorflow/issues/4078


# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn8=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers

# Install Python
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install future typing
RUN pip install numpy \
        keras \
        h5py

RUN pip install ${TENSORFLOW_PIP}==${TENSORFLOW_VERSION}
RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION}

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, no CUDA stubs needed because we set LD_LIBRARY_PATH
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod==${HOROVOD_VERSION}

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Download examples
RUN apt-get install -y --no-install-recommends subversion && \
    svn checkout https://github.com/horovod/horovod/trunk/examples && \
    rm -rf /examples/.svn

WORKDIR "/examples"

###### Modifications to horovod Dockerfile below
# tensorflow_addons is tightly coupled to TF version. TF 2.1 = 0.9.1, TF 2.2 = 0.10.0
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        scikit-learn==0.23.1 \
        wandb==0.9.1 \
        tensorboard_plugin_profile \
        tensorflow-addons==${TENSORFLOW_ADDONS_VERSION} \
        colorama==0.4.3 \
        pandas \
        apache_beam

ENV HDF5_USE_FILE_LOCKING "FALSE"

WORKDIR /fsx
CMD ["/bin/bash"]

###### Modifications specifically for SageMaker are below
# Install SSH on SageMaker machines
RUN apt-get install -y --no-install-recommends openssh-client openssh-server
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN mkdir -p /root/.ssh/ && \
    mkdir -p /var/run/sshd && \
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

RUN pip install --no-cache-dir \
    mpi4py==3.0.3 \
    sagemaker-training==3.7.2

 RUN pip install --no-cache-dir \
    transformers==4.2.0 \
    datasets==1.2.1 \
    tokenizers==0.9.4 \
    sentencepiece==0.1.95

###### Modifications specifically for EC2 connected to FSx for Lustre are below
# When you use `docker run`, you'll need to run two commands manually:
# pip install -e /fsx/transformers
# These are done in the MPIJob launch script when using Kubernetes, but not for a shell.
