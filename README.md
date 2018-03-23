# NLP-Cube

# Setup:

Before running the server, you need the model's weights, and you can follow two approaches to get them:
* Download data in order to train the model yourself
* Download already existing model weights 


#### Installing dyNET:

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
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DMKL_ROOT=/opt/intel/mkl -DPYTHON=`which python2`

    make -j 2 # replace 2 with the number of available cores
    make install

    cd python
    python2 ../../setup.py build --build-dir=.. --skip-build install
    ```
    

#### Training the lemmatizer:

Use the following command to train your lemmatizer:

```python2 cube/main.py --train=lemmatizer --train-file=corpus/ud_treebanks/UD_Romanian/ro-ud-train.conllu --dev-file=corpus/ud_treebanks/UD_Romanian/ro-ud-dev.conllu --embeddings=corpus/wiki.ro.vec --store=corpus/trained_models/ro/lemma/lemma --test-file=corpus/ud_test/gold/conll17-ud-test-2017-05-09/ro.conllu --batch-size=1000```

#### Running the server:

Use the following command to run the server locally:

```python2 cube/main.py --start-server --model-tokenization=corpus/trained_models/ro/tokenizer --model-parsing=corpus/trained_models/ro/parser --model-lemmatization=corpus/trained_models/ro/lemma --embeddings=corpus/wiki.ro.vec --server-port=8080```


# Current status
* we treat words and character embeddings in a similar fashion 
* we tested with character encodings only (feature cutoff is set at 100)

# ToDO
- [ ] provide training examples
- [x] add word embeddings
- [x] find a good network achitecture for POS tagging
- [x] prepare a neural/based language pipeline
- [ ] pre-train models using universal dependencies
- [x] add a parser

# Parser architecture
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

