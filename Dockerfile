ARG IMAGE_NAME
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 as base

FROM base as base-amd64

ENV NV_CUDNN_VERSION 8.4.0.27
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.6"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda11.6"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         locales \
         cmake \
         git \
         curl \
         vim \
         unzip \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libfreetype6-dev \
         libxft-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda116 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
RUN conda install -c conda-forge tensorflow tensorflow=2.9.1

RUN pip install --upgrade pip
RUN pip install tqdm==4.64.0 &&\
     pip install matplotlib==3.5.2 &&\
     pip install seaborn==0.11.2

WORKDIR /workspace
RUN mkdir exp_src
WORKDIR  exp_src
