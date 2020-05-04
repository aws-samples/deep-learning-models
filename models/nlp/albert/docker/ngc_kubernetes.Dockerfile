# Contains TensorFlow 2.1 and Horovod 0.19.0
FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

RUN apt-get update && \
    apt-get install -y iputils-ping net-tools openmpi-bin clustershell htop

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        scikit-learn \
        gputil \
        requests \
        tensorflow-addons==0.9.1
# TODO: Why does installing torch break TF XLA support?

ENV HDF5_USE_FILE_LOCKING "FALSE"

WORKDIR /fsx/model-optimization
CMD ["/bin/bash"]
