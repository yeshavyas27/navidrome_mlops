FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.12

USER root

# 1. Install system tools and gdown for Google Drive downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvtop tar gzip wget gnupg build-essential python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install gdown specifically for the large Drive file
RUN python3 -m pip install --no-cache-dir gdown

# 2. Latest CUDA Toolkit for H100 (Hopper)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8 && \
    rm cuda-keyring_1.1-1_all.deb 

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /home/${NB_USER}/work

# 3. Download Dataset from Google Drive
# The ID is extracted from your link: 1n2aaVgfUgz_8Uif4hCmI5meyX4jDhI6u
RUN gdown 1n2aaVgfUgz_8Uif4hCmI5meyX4jDhI6u -O ThirtyMusic.tar.gz

# 4. Extract Dataset
RUN mkdir -p data/ThirtyMusic && \
    tar -xzf ThirtyMusic.tar.gz -C data/ThirtyMusic --strip-components=1 && \
    rm ThirtyMusic.tar.gz

# 5. Copy the rest of the Repo (Scripts, Requirements, etc.)
COPY . .

# 6. Install Dependencies
RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools && \
    python -m pip install --no-cache-dir -r train/requirements.txt

USER ${NB_UID}

ENV MLFLOW_TRACKING_URI="http://129.114.27.248:8000"

# Fix permissions
RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"