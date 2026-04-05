# Use a Jupyter base notebook pinned to Ubuntu 22.04 (Jammy) 
# to ensure compatibility with the AMD ROCm repository
FROM quay.io/jupyter/base-notebook:ubuntu-22.04

# Define ROCm arguments from your reference
ARG ROCM_VERSION=7.0.1
ARG AMDGPU_VERSION=7.0.1.70001

USER root

# 1. Install system prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    tar \
    gzip \
    wget \
    gnupg \
    build-essential \
    cmake \
    libncurses-dev \
    libsystemd-dev \
    libudev-dev \
    libdrm-dev \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. Install AMD ROCm system stack (from your reference)
RUN wget https://repo.radeon.com/amdgpu-install/$ROCM_VERSION/ubuntu/jammy/amdgpu-install_$AMDGPU_VERSION-1_all.deb && \
    apt-get update --yes && \
    apt-get install --yes ./amdgpu-install_$AMDGPU_VERSION-1_all.deb && \
    amdgpu-install --usecase=rocm -y && \
    rm *.deb

# 3. Build nvtop from source with AMD support for monitoring
RUN git clone https://github.com/Syllo/nvtop && \
    mkdir -p nvtop/build && cd nvtop/build && \
    cmake -DAMDGPU_SUPPORT=ON -DNVIDIA_SUPPORT=OFF .. && \
    make && make install && \
    cd ../.. && rm -rf nvtop

# 4. Install PyTorch with ROCm (AMD) compatibility and Lightning
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/rocm7.0 \
    'torch' \
    'torchaudio' \
    'torchvision'

RUN python3 -m pip install --no-cache-dir 'lightning'

# 5. Define the working directory inside the container
WORKDIR /home/${NB_USER}/work

# 6. Copy your entire repository context
COPY . .

# 7. Install your project's Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip wheel setuptools && \
    python3 -m pip install --no-cache-dir -r train/requirements.txt && \
    python3 -m pip install --no-cache-dir bash_kernel && \
    python3 -m bash_kernel.install --sys-prefix

# 8. CRITICAL: Fix permissions as ROOT before switching users
RUN chown -R ${NB_UID}:${NB_GID} /home/${NB_USER}/work && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# 9. Switch to the non-root user for security and Jupyter standards
USER ${NB_UID}

# Set the MLflow Tracking URI
ENV MLFLOW_TRACKING_URI="http://129.114.27.248:8000"

# Default command to start Jupyter
CMD ["start-notebook.sh", "--NotebookApp.token=''"]