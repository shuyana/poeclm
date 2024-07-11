FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:$PATH"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV DEBIAN_FRONTEND="noninteractive"

SHELL ["/bin/bash", "-c"]

ENV LANG="en_US.UTF-8"
ENV LANGUAGE="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        language-pack-en \
        locales \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        software-properties-common \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        git \
        gosu \
        htop \
        less \
        libxext6 \
        libxrender1 \
        nvtop \
        openbabel \
        python-is-python3 \
        python3.11 \
        python3.11-dev \
        vim \
        wget \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11

ENV PATH="/home/user/.local/bin:$PATH"
RUN useradd -m -s /bin/bash user \
    && curl -fsSL https://install.python-poetry.org | gosu user python3 -
COPY --chmod=755 <<EOF /usr/local/bin/entrypoint.sh
#!/bin/bash
groupmod -g \${GID:-9001} -o user &>/dev/null
usermod -u \${UID:-9001} -d /home/user -m -o user &>/dev/null
exec gosu user "\$@"
EOF
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
