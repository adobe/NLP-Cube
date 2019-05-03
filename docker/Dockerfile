FROM ubuntu


# Installing build dependencies
RUN apt-get update && apt-get install -y build-essential automake make cmake g++ wget git mercurial python3-pip curl

# Preparing Python build environment
RUN pip3 install cython future scipy nltk requests xmltodict nose2

# Installing MKL library
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list && \
    apt-get update && \
    apt-get install -y intel-mkl-64bit-2018.2-046

# Installing DyNET
RUN mkdir dynet-base && \
    cd dynet-base && \
    git clone https://github.com/clab/dynet.git && \
    hg clone https://bitbucket.org/eigen/eigen -r b2e267d && \
    cd dynet && \
    mkdir build && \
    cd build && \
    cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=/usr/bin/python3 -DMKL_ROOT=/opt/intel/mkl && \
    make -j 2 && \
    cd python && \
    python3 ../../setup.py build --build-dir=.. --skip-build install

# Prepare environment UTF-8
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Prepare cube

RUN mkdir /work && \
    cd /work && \
    git clone https://github.com/adobe/NLP-Cube.git

# Prepare notebook
RUN pip3 install jupyter

# Start notebook
CMD cd /work/NLP-Cube/ && python3 -m "notebook" --allow-root --ip=0.0.0.0 --no-browser

