FROM quay.io/toolbx-images/ubuntu-toolbox:22.04

ARG ROCM_VERSION=7.0.1
ARG AMDGPU_VERSION=7.0.1.70001
ARG DEBIAN_FRONTEND=noninteractive

USER root

# 1. System prereqs
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. ROCm stack
RUN wget https://repo.radeon.com/amdgpu-install/$ROCM_VERSION/ubuntu/jammy/amdgpu-install_$AMDGPU_VERSION-1_all.deb && \
    apt-get update --yes && \
    apt-get install --yes ./amdgpu-install_$AMDGPU_VERSION-1_all.deb && \
    amdgpu-install --usecase=rocm --no-dkms -y && \
    rm *.deb && \
    rm -rf /var/lib/apt/lists/*

# 3. PyTorch ROCm
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/rocm7.0 torch

# 4. Training deps
RUN pip3 install --no-cache-dir \
    mlflow \
    pandas \
    numpy \
    tqdm \
    optuna \
    psutil

# 5. Non-root user
RUN useradd -m -s /bin/bash appuser
WORKDIR /home/appuser/work
COPY train/ ./train/
RUN chown -R appuser:appuser /home/appuser/work
USER appuser

ENV MLFLOW_TRACKING_URI="http://129.114.25.168:8000"
