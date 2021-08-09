![Monthly](https://img.shields.io/pypi/dm/nlpcube.svg) ![Weekly](https://img.shields.io/pypi/dw/nlpcube.svg) ![daily](https://img.shields.io/pypi/dd/nlpcube.svg)
![Version](https://badge.fury.io/py/nlpcube.svg) [![Python 3](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/release/python-360/) 

## News
**[05 August 2021]** - We are releasing version 3.0 of NLPCube and models and introducing [FLAVOURS](#flavours). This is a major update, but we did our best to maintain the same API, so previous implementation will not crash. The supported language list is smaller, but you can open an issue for unsupported languages, and we will do our best to add them. Other options include fixing the pip package version 1.0.8 ```pip install nlpcube==0.1.0.8```. 

**[15 April 2019]** - We are releasing version 1.1 models - check all [supported languages below](#languages). Both models 1.0 and 1.1 are trained on the same UD2.2 corpus; however, models 1.1 do not use vector embeddings, thus reducing the time and disk space required to download them. Some languages actually have a slightly increased accuracy, some a bit decreased. By default, NLP Cube will use the latest (at this time) 1.1 models.

To use the older 1.0 models just specify this version in the ``load`` call: ``cube.load("en",1.0)`` (``en`` for English, or any other language code). This will download (if not already downloaded) and use _this_ specific model version. Same goes for any language/version you want to use.

If you already have NLP Cube installed and **want to use the newer 1.1 models**, type either ``cube.load("en",1.1)`` or ``cube.load("en","latest")`` to auto-download them. After this, calling ``cube.load("en")`` without version number will automatically use the latest ones from your disk.

<hr>

# NLP-Cube

NLP-Cube is an opensource Natural Language Processing Framework with support for languages which are included in the [UD Treebanks](http://universaldependencies.org/) (list of all available languages below). Use NLP-Cube if you need:
* Sentence segmentation
* Tokenization
* POS Tagging (both language independent (UPOSes) and language dependent (XPOSes and ATTRs))
* Lemmatization
* Dependency parsing

Example input: **"This is a test."**, output is: 
```
1       This    this    PRON    DT      Number=Sing|PronType=Dem        4       nsubj   _
2       is      be      AUX     VBZ     Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin   4       cop     _
3       a       a       DET     DT      Definite=Ind|PronType=Art       4       det     _
4       test    test    NOUN    NN      Number=Sing     0       root    SpaceAfter=No
5       .       .       PUNCT   .       _       4       punct   SpaceAfter=No
```

**If you just want to run it**, here's how to set it up and use NLP-Cube in a few lines: [Quick Start Tutorial](examples/1.%20NLP-Cube%20Quick%20Tutorial.ipynb).

For **advanced users that want to create and train their own models**, please see the Advanced Tutorials in ``examples/``, starting with how to [locally install NLP-Cube](examples/2.%20Advanced%20usage%20-%20NLP-Cube%20local%20installation.ipynb).

## Simple (PIP) installation / update version

Install (or update) NLP-Cube with:

```bash
pip3 install -U nlpcube
```
### API Usage 

To use NLP-Cube ***programmatically** (in Python), follow [this tutorial](examples/1.%20NLP-Cube%20Quick%20Tutorial.ipynb)
The summary would be:
```python
from cube.api import Cube       # import the Cube object
cube=Cube(verbose=True)         # initialize it
cube.load("en", device='cpu')   # select the desired language (it will auto-download the model on first run)
text="This is the text I want segmented, tokenized, lemmatized and annotated with POS and dependencies."
document=cube(text)            # call with your own text (string) to obtain the annotations
```
The ``document`` object now contains the annotated text, one sentence at a time. To print the third words's POS (in the first sentence), just run:
```
print(document.sentences[0][2].upos) # [0] is the first sentence and [2] is the third word
```
Each token object has the following attributes: ``index``, ``word``, ``lemma``, ``upos``, ``xpos``, ``attrs``, ``head``, ``label``, ``deps``, ``space_after``. For detailed info about each attribute please see the standard CoNLL format.

### Flavours

Previous versions on NLP-Cube were trained on individual treebanks. This means that the same language was supported by 
multiple models at the same time. For instance, you could parse English (en) text with `en_ewt`, `en_esl`, `en_lines`, 
etc. The current version of NLPCube combines all flavours of a treebank under the same umbrella, by jointly optimizing
a conditioned model. You only need to load the base language, for example `en` and then select which flavour to apply
at runtime:

```text
from cube.api import Cube       # import the Cube object
cube=Cube(verbose=True)         # initialize it
cube.load("en", device='cpu')   # select the desired language (it will auto-download the model on first run)
text="This is the text I want segmented, tokenized, lemmatized and annotated with POS and dependencies."


# Parse using the default flavour (in this case EWT)
document=cube(text)            # call with your own text (string) to obtain the annotations
# or you can specify a flavour
document=cube(text, flavour='en_lines') 
```

### Webserver Usage 
The current version dropped supported, since most people preferred to implement their one NLPCube as a service.

## Cite

If you use NLP-Cube in your research we would be grateful if you would cite the following paper: 
* [**NLP-Cube: End-to-End Raw Text Processing With Neural Networks**](http://www.aclweb.org/anthology/K18-2017), Boroș, Tiberiu and Dumitrescu, Stefan Daniel and Burtica, Ruxandra, Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies, Association for Computational Linguistics. p. 171--179. October 2018 

or, in bibtex format: 

```
@InProceedings{boro-dumitrescu-burtica:2018:K18-2,
  author    = {Boroș, Tiberiu  and  Dumitrescu, Stefan Daniel  and  Burtica, Ruxandra},
  title     = {{NLP}-Cube: End-to-End Raw Text Processing With Neural Networks},
  booktitle = {Proceedings of the {CoNLL} 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
  month     = {October},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  pages     = {171--179},
  abstract  = {We introduce NLP-Cube: an end-to-end Natural Language Processing framework, evaluated in CoNLL's "Multilingual Parsing from Raw Text to Universal Dependencies 2018" Shared Task. It performs sentence splitting, tokenization, compound word expansion, lemmatization, tagging and parsing. Based entirely on recurrent neural networks, written in Python, this ready-to-use open source system is freely available on GitHub. For each task we describe and discuss its specific network architecture, closing with an overview on the results obtained in the competition.},
  url       = {http://www.aclweb.org/anthology/K18-2017}
}
```


## Languages and performance

For comparison, the performance of 3.0 models is reported on the 2.2 UD corpus, but distributed models are obtained from UD 2.7.

Results are reported against the test files for each language (available in the UD 2.2 corpus) using the 2018 conll eval script. Please see more info about what [each metric represents here](http://universaldependencies.org/conll18/evaluation.html). 

Notes: 
* version 1.1 of the models no longer need the large external vector embedding files. This makes loading the 1.1 models faster and less RAM-intensive.
* all reported results here are end-2-end. (e.g. we test the tagging accuracy on our own segmented text, as this is the real use-case; CoNLL results are mostly reported on "gold" - or pre-segmented text, leading to higher accuracy for the tagger/parser/etc.)

|Language|Model|Token|Sentence|UPOS|XPOS|AllTags|Lemmas|UAS|LAS|
|--------|-----|:---:|:------:|:--:|:--:|:-----:|:----:|:-:|:-:|
|Chinese|
| |zh-1.0|93.03|99.10|88.22|88.15|86.91|92.74|73.43|69.52|
| |zh-1.1|92.34|99.10|86.75|86.66|85.35|92.05|71.00|67.04|
| |zh.3.0|95.88|87.36|91.67|83.54|82.74|85.88|79.15|70.08|
|English|
| |en-1.0|99.25|72.8|95.34|94.83|92.48|95.62|84.7|81.93|
| |en-1.1|99.2|70.94|94.4|93.93|91.04|95.18|83.3|80.32|
| |en-3.0|98.61|71.92|96.01|95.71|93.75|96.06|85.40|82.72|
|Hungarian|
| |hu-1.0|99.8|94.18|94.52|99.8|86.22|91.07|81.57|75.95|
| |hu-1.1|99.88|97.77|93.11|99.88|86.79|91.18|77.89|70.94|
| |hu-3.0|99.75|91.64|96.43|99.75|89.89|91.31|86.34|81.29|
|Romanian (RO-RRT)|
| |ro-1.0|99.74|95.56|97.42|96.59|95.49|96.91|90.38|85.23|
| |ro-1.1|99.71|95.42|96.96|96.32|94.98|96.57|90.14|85.06|
| |ro-3.0|99.80|95.64|97.67|97.11|96.76|97.55|92.06|87.67|
|Spanish|
| |es-1.0|99.98|98.32|98.0|98.0|96.62|98.05|90.53|88.27|
| |es-1.1|99.98|98.40|98.01|98.00|96.6|97.99|90.51|88.16|
| |es-3.0|99.96|97.17|94.47|96.60|91.43|94.82|87.97|85.72|


