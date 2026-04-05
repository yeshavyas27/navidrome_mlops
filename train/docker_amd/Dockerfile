# Use a standard Jupyter base notebook (no CUDA)
FROM quay.io/jupyter/base-notebook:python-3.12

USER root

# 1. Install system dependencies for extraction and AMD GPU monitoring
# We must build nvtop from source because older versions in Ubuntu 
# package repositories do not support AMD GPUs.
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

# Build nvtop from source with AMD support
RUN git clone https://github.com/Syllo/nvtop && \
    mkdir -p nvtop/build && cd nvtop/build && \
    cmake -DAMDGPU_SUPPORT=ON -DNVIDIA_SUPPORT=OFF .. && \
    make && make install && \
    cd ../.. && rm -rf nvtop

# 2. Install PyTorch with ROCm (AMD) support instead of CUDA
# Using the rocm7.0 wheel index
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/rocm7.0 \
    'torch' \
    'torchaudio' \
    'torchvision'

# 3. Define the working directory inside the container
WORKDIR /home/${NB_USER}/work

# 4. Copy your entire repository context
COPY . .

# 5. Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools && \
    python -m pip install --no-cache-dir -r train/requirements.txt && \
    python -m pip install --no-cache-dir bash_kernel && \
    python -m bash_kernel.install --sys-prefix

# 6. CRITICAL: Fix permissions as ROOT before switching users
# This prevents the exit code: 1 error
RUN chown -R ${NB_UID}:${NB_GID} /home/${NB_USER}/work && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# 7. Switch to the non-root user for security and Jupyter standards
USER ${NB_UID}

# Set the MLflow Tracking URI
ENV MLFLOW_TRACKING_URI="http://129.114.27.248:8000"

# Default command to start Jupyter
CMD ["start-notebook.sh", "--NotebookApp.token=''"]