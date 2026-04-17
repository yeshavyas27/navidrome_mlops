FROM rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1

ARG DEBIAN_FRONTEND=noninteractive

# ROCm + PyTorch + Python 3.10 already in base image
# Just install your pip deps
RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    numpy \
    tqdm \
    optuna \
    psutil

# Non-root user
RUN useradd -m -s /bin/bash appuser
WORKDIR /home/appuser/work
COPY train/ ./train/
RUN chown -R appuser:appuser /home/appuser/work
USER appuser

ENV MLFLOW_TRACKING_URI="http://129.114.25.168:8000"
