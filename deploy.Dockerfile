FROM nvcr.io/nvidia/pytorch:22.12-py3
# ARG HUGGING_FACE_HUB_TOKEN
ENV DEBIAN_FRONTEND noninteractive
# gradio and streamlit default ports
# EXPOSE 7860 8501
# EXPOSE 7500-8999

### Set default shell to /bin/bash ###
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Install some basic utilities
RUN apt-get update && apt-get install python3 python3-pip -y

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    wget \
    unzip \
    nvidia-cuda-toolkit \
 && rm -rf /var/lib/apt/lists/*

### Set default NCCL parameters ###
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# WORKDIR /home/user
# WORKDIR /home/user/app
### Mount Point ###
# When launching the container, mount the code directory to /app
ARG APP_MOUNT=/app
VOLUME [ ${APP_MOUNT} ]
WORKDIR ${APP_MOUNT}

# RUN pip3 install torchvision torchaudio torch --index-url https://download.pytorch.org/whl/cu118
COPY ./requirements.docker.txt /requirements.docker.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install --no-cache-dir -r /requirements.docker.txt

### Create a non-root user ###
# https://github.com/facebookresearch/detectron2/blob/v0.3/docker/Dockerfile
# https://code.visualstudio.com/docs/remote/containers-advanced#_creating-a-nonroot-user
ARG USER=appuser
ARG UID=1000
ARG GID=1000
RUN useradd -m --no-log-init --system  --uid ${UID} ${USER} -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
ENV PATH=/home/${USER}/.local/bin:${PATH}
RUN chown -R ${UID}:${GID} /home/${USER} \
    && chown -R ${UID}:${GID} /usr/local/lib/python* \
    && chown -R ${UID}:${GID} /usr/lib/python*
USER ${USER}
ENTRYPOINT ["/bin/bash"]