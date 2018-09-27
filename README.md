# NLP-Cube

NLP-Cube is an opensource Natural Language Processing Framework with support for languages which are included in the [UD Treebanks](http://universaldependencies.org/). 

Follow the [Quick Start Tutorial](https://github.com/adobe/NLP-Cube/blob/pip3.package/examples/simple_example.ipynb) to get things running in no time.

Advanced users that want to create their own models, will have to use the installation tutorial (below).

## Simple (PIP) installation

If you just want to use NLP-Cube, just use the available PIP package:

```bash
pip3 install nlpcube
```
### Usage

To use NLP-Cube programmatically (in Python), follow [this tutorial](https://github.com/adobe/NLP-Cube/blob/pip3.package/examples/simple_example.ipynb)

To use NLP-Cube as a web service, you need to clone this repo, install requirements and start the server:

```bash
git clone https://github.com/adobe/NLP-Cube.git
cd NLP-Cube
pip3 install -r requirements.txt
```
The following command will start the server and preload languages: en, fr and de.
```bash
cd cube
python3 webserver.py --port 8080 --lang=en --lang=fr --lang=de
``` 

To test, open the following [this link](http://localhost:8080/nlp?lang=en&text=This%20is%20a%20simple%20test)


## Manual Installation (if you want to train new models)

### Cloning NLP-Cube

In order to create new models you need to start by cloning this repo and installing requirements.

**Clone**
```bash
git clone https://github.com/adobe/NLP-Cube.git
cd NLP-Cube
pip3 install -r requirements.txt
```

NLP-Cube is dependent on [DyNET](https://github.com/clab/dynet). In order to train your own models you should do a custom DyNET installation with MKL and/or CUDA support.


### Installing DyNet:

1. Make sure you have [Mercurial](https://www.mercurial-scm.org/wiki/Download), [python](https://www.python.org/downloads/), [pip](https://pip.pypa.io/en/stable/installing/), [cmake](https://cmake.org/install/) installed (you can also check steps documented [here](http://dynet.readthedocs.io/en/latest/python.html#installing-a-cutting-edge-and-or-gpu-version))
2. [Hard mode] Install Intel's [MKL](https://software.seek.intel.com/performance-libraries) library. Download appropriate version for your OS and follow the install script provided in the archive. MKL is a optimized math library that `DyNet` can use to significantly speed up training and runtime performance.

OR

2. [Easy mode] If you run a debian (should work on other \*nix systems), run the following commands to automatically setup MKL:
```bash
sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB 
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
sudo apt-get update 
sudo apt-get install -y intel-mkl-64bit-2018.2-046
```

OR

2. [Don't really care about speed mode] Do not install MKL at all. This will slow down `DyNet` by ~2.5 times but it will work just as well. Don't forget to run cmake in step 3. without the "-DMKL_ROOT=/opt/intel/mkl" flag in this case.

3. Install `DyNet` by using the installation steps from the [manual installation page](http://dynet.readthedocs.io/en/latest/python.html#manual-installation). More specifically, you should use:

    ```
    pip install cython
    mkdir dynet-base
    cd dynet-base

    git clone https://github.com/clab/dynet.git
    hg clone https://bitbucket.org/eigen/eigen -r 2355b22  # -r NUM specified a known working revision

    cd dynet
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DMKL_ROOT=/opt/intel/mkl -DPYTHON=`which python3`

    make -j 2 # replace 2 with the number of available cores
    make install

    cd python
    python3 ../../setup.py build --build-dir=.. --skip-build install
    ```

Note: sometimes cmake fails. If it does, delete the contents of the build folder and give the -DEIGEN3_INCLUDE_DIR flag the absolute path to eigen (dont use ../ or other relative paths). Also, check cmake is updated to the latest version available. 

### Training

Training models is easy. Just use `--help` command line to get available command. Depending on what model you want to train, you must set the appropiate value for the `--train` parameter. For example, if you want to train the lemmatizer, you need to use the following command (provided that you have downloaded the training data and placed it in the `corpus` folder:

```bash
python=3 cube/main.py --train=lemmatizer --train-file=corpus/ud_treebanks/UD_Romanian/ro-ud-train.conllu --dev-file=corpus/ud_treebanks/UD_Romanian/ro-ud-dev.conllu --embeddings=corpus/wiki.ro.vec --store=corpus/trained_models/ro/lemma/lemma --test-file=corpus/ud_test/gold/conll17-ud-test-2017-05-09/ro.conllu --batch-size=1000
```
