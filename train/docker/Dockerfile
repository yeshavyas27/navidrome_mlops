FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.12
# builds on https://github.com/jupyter/docker-stacks/blob/main/images/pytorch-notebook/cuda12/Dockerfile

USER root

# nvtop is for monitoring GPU utilization
RUN apt update && apt install nvtop

# need CUDA toolkit to install flash attention and deepspeed
RUN apt-get update && apt-get install -y wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8 && \
    rm cuda-keyring_1.1-1_all.deb 


ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

USER ${NB_UID}

RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools

RUN python -m pip install --no-cache-dir bash_kernel && \
    python -m bash_kernel.install --sys-prefix

# Install Lightning and Transformers
RUN python -m pip install --pre --no-cache-dir lightning transformers datasets accelerate sentencepiece bitsandbytes peft && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Download million song dataset subset 10k song
RUN wget http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz
RUN tar -xzf millionsongsubset.tar.gz && \
    rm millionsongsubset.tar.gz

# Deepspeed for offloading computation to CPU for very large models
RUN pip install --pre --no-cache-dir deepspeed && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"