# NLP-Cube

NLP-Cube is an opensource Natural Language Processing Framework with support for languages which are included in the [UD Treebanks](http://universaldependencies.org/). 

Follow the [Quick Start Tutorial](https://github.com/adobe/NLP-Cube/blob/pip3.package/examples/simple_example.ipynb) to get things running in no time.

Advanced users that want to create their own models, will have to use the installation tutorial (below).


## Installation

NLP-Cube is dependent on [DyNET](https://github.com/clab/dynet). In order to train your own models you should do a custom DyNET installation with MKL and/or CUDA support.


### Installing dyNET:

1. Make sure you have [Mercurial](https://www.mercurial-scm.org/wiki/Download), [python](https://www.python.org/downloads/), [pip](https://pip.pypa.io/en/stable/installing/), [cmake](https://cmake.org/install/) installed (you can also check steps documented [here](http://dynet.readthedocs.io/en/latest/python.html#installing-a-cutting-edge-and-or-gpu-version))
2. Install Intel's [MKL](https://software.seek.intel.com/performance-libraries) library
3. Install `dyNET` by using the installation steps from the [manual installation page](http://dynet.readthedocs.io/en/latest/python.html#manual-installation). More specifically, you should use:

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
    
### Cloning or installing cube

In order to install NLP-Cube, you can clone this repo (to get the latest version) or just use the pip package (to get the latest stable version):

**Clone**
```bash
git clone https://github.com/adobe/NLP-Cube.git
```

**PIP**
```bash
pip3 install --upgrade nlpcube
```

### Training

Training models is easy. Just use `--help` command line to get available command. Depending on what model you want to train, you must set the appropiate value for the `--train` parameter. For example, if you want to train the lemmatizer, you need to use the following command (provided that you have downloaded the training data and placed it in the `corpus` folder:

```bash
python=3 cube/main.py --train=lemmatizer --train-file=corpus/ud_treebanks/UD_Romanian/ro-ud-train.conllu --dev-file=corpus/ud_treebanks/UD_Romanian/ro-ud-dev.conllu --embeddings=corpus/wiki.ro.vec --store=corpus/trained_models/ro/lemma/lemma --test-file=corpus/ud_test/gold/conll17-ud-test-2017-05-09/ro.conllu --batch-size=1000
```

#### Running the server:

Use the following command to run the server locally:

```python3 cube/main.py --start-server --model-tokenization=corpus/trained_models/ro/tokenizer --model-parsing=corpus/trained_models/ro/parser --model-lemmatization=corpus/trained_models/ro/lemma --embeddings=corpus/wiki.ro.vec --server-port=8080```


## Parser architecture
```
#   -----------------                    -------------------------- 
#   |word emebddings|----          ------|morphological embeddings|
#   -----------------    |        |      --------------------------
#                        |        |
#                      --------------
#                      |concatenate |
#                      --------------
#                             |
#                     ----------------
#                     |bdlstm_1_layer|
#                     ----------------
#                             |
#                     ----------------                  
#                     |bdlstm_2_layer| 
#                     ----------------                    
#                             |-----------------------------------------------------------------                          
#                     ----------------                                                         |
#                     |bdlstm_3_layer|                                                         |
#                     ----------------                                                         |
#                             |                                                                |
#        ---------------------------------------------                    ---------------------------------------------              
#        |           |                |              |                    |           |                |              |
#        |           |                |              |                    |           |                |              |
#    ---------  -----------       ----------    ------------          ---------  -----------       ----------    ------------
#    |to_link|  |from_link|       |to_label|    |from_label|          |to_link|  |from_link|       |to_label|    |from_label|
#    ---------  -----------       ----------    ------------          ---------  -----------       ----------    ------------
#         |        |                      |       |                       |           |                  |            |
#       --------------                 ---------------                  ------------------            -------------------
#       |softmax link|                 |softmax label|                  |aux softmax link|            |aux softmax label|
#       --------------                 ---------------                  ------------------            -------------------
#
#

```

# Tagger architecture

```
#   -----------------                    ---------------------- 
#   |word emebddings|----          ------|character embeddings|
#   -----------------    |        |      ----------------------
#                        |        |
#                      --------------
#                      |tanh_1_layer|
#                      --------------
#                             |
#                     ----------------
#                     |bdlstm_1_layer|
#                     ----------------
#                             |
#                      --------------                  
#                      |tanh_2_layer|-------------------
#                      --------------                   |
#                             |                         |
#                     ----------------         -------------------
#                     |bdlstm_2_layer|         |aux_softmax_layer|
#                     ----------------         -------------------
#                             |
#                      ---------------
#                      |softmax_layer|
#                      ---------------
#

```

