# Contains TensorFlow 2.1 and Horovod 0.19.0
FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y python3-opencv libopenblas-base \
	libomp-dev build-essential

# The minor version needs to be bumped with every NGC upgrade
ENV PATH="/usr/local/nvm/versions/node/v13.11.0/bin:${PATH}"

RUN pip install ipywidgets \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN pip install jupyterlab-nvdashboard tqdm tensorflow_datasets \
    && jupyter labextension install jupyterlab-nvdashboard

RUN pip install matplotlib scikit-learn scikit-image seaborn \
	cython numba tqdm tensorflow_addons tensorflow_datasets \
	&& pip install pycocotools

CMD nohup jupyter lab --allow-root --ip=0.0.0.0 --no-browser > notebook.log

# docker run -it --rm -d --gpus all --net=host --name tensorflow -v ~/workspace:/workspace/shared_workspace jarednielsen/albert-tf:ngc_gpumonitor
# ssh -L localhost:6006:localhost:8888 ubuntu@${IP_FSX_CONNECTION}
# docker exec tensorflow bash -c "jupyter notebook list"
