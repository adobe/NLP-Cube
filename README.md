[![Downloads](https://pepy.tech/badge/nlpcube)](https://pepy.tech/project/nlpcube) [![Downloads](https://pepy.tech/badge/nlpcube/month)](https://pepy.tech/project/nlpcube/month) ![Weekly](https://img.shields.io/pypi/dw/nlpcube.svg) ![daily](https://img.shields.io/pypi/dd/nlpcube.svg)
![Version](https://badge.fury.io/py/nlpcube.svg) [![Python 3](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/release/python-360/) 

## News

**[15 April 2019]** - We are releasing version 1.1 models - check all [supported languages below](#languages). Both 1.0 and 1.1 models are trained on the same [UD2.2 corpus](http://hdl.handle.net/11234/1-2837); however, models 1.1 do not use vector embeddings, thus reducing disk space and time required to use them. Some languages actually have a slightly increased accuracy, some a bit decreased. By default, NLP Cube will use the latest (at this time) 1.1 models.

To use the older 1.0 models just specify this version in the ``load`` call: ``cube.load("en", 1.0)`` (``en`` for English, or any other language code). This will download (if not already downloaded) and use _this_ specific model version. Same goes for any language/version you want to use.

If you already have NLP Cube installed and **want to use the newer 1.1 models**, type either ``cube.load("en", 1.1)`` or ``cube.load("en", "latest")`` to auto-download them. After this, calling ``cube.load("en")`` without version number will automatically use the latest ones from your disk.

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
```
from cube.api import Cube       # import the Cube object
cube=Cube(verbose=True)         # initialize it
cube.load("en")                 # select the desired language (it will auto-download the model on first run)
text="This is the text I want segmented, tokenized, lemmatized and annotated with POS and dependencies."
sentences=cube(text)            # call with your own text (string) to obtain the annotations
```
The ``sentences`` object now contains the annotated text, one sentence at a time. To print the third word's POS (in the first sentence), just run:
```
print(sentences[0][2].upos) # [0] is the first sentence and [2] is the third word
```
Each token object has the following attributes: ``index``, ``word``, ``lemma``, ``upos``, ``xpos``, ``attrs``, ``head``, ``label``, ``deps``, ``space_after``. For detailed info about each attribute please see the standard CoNLL format.

### Webserver Usage 

To use NLP-Cube as a **web service**, you need to 
[locally install NLP-Cube](examples/2.%20Advanced%20usage%20-%20NLP-Cube%20local%20installation.ipynb) 
and start the server:

For example, the following command will start the server and preload languages: en, fr and de.
```bash
cd cube
python3 webserver.py --port 8080 --lang=en --lang=fr --lang=de
``` 

To test, open the following [link](http://localhost:8080/nlp?lang=en&text=This%20is%20a%20simple%20test) (please copy the address of the link as it is a local address and port link)

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


## <a name="languages"></a>Languages and performance

Results are reported against the test files for each language (available in the UD 2.2 corpus) using the 2018 conll eval script. Please see more info about what [each metric represents here](http://universaldependencies.org/conll18/evaluation.html). 

Notes: 
* version 1.1 of the models no longer need the large external vector embedding files. This makes loading the 1.1 models faster and less RAM-intensive.
* all reported results here are end-2-end. (e.g. we test the tagging accuracy on our own segmented text, as this is the real use-case; CoNLL results are mostly reported on "gold" - or pre-segmented text, leading to higher accuracy for the tagger/parser/etc.)

|Language|Model|Token|Sentence|UPOS|XPOS|AllTags|Lemmas|UAS|LAS|
|--------|-----|:---:|:------:|:--:|:--:|:-----:|:----:|:-:|:-:|
|Afrikaans|
| |af-1.0|99.97|99.65|97.28|93.0|91.53|96.42|87.61|83.96|
| |af-1.1|99.99|99.29|96.72|92.29|90.87|96.48|87.32|83.31|
|Ancient Greek|
| |grc-1.0|100.0|18.13|94.92|95.32|84.17|86.59|72.44|67.73|
| |grc-1.1|100.0|17.61|96.87|97.35|88.36|88.41|73.4|69.36|
|Arabic|
| |ar-1.0|99.98|61.05|73.42|69.75|68.12|41.26|53.94|50.31|
| |ar-1.1|99.99|60.53|73.27|68.98|65.95|40.87|53.06|49.45|
|Armenian|
| |hy-1.0|97.34|87.52|74.13|96.76|41.51|60.58|11.41|1.7|
|Basque|
| |eu-1.0|99.97|99.83|94.93|99.97|87.24|90.75|85.49|81.35|
| |eu-1.1|99.97|99.75|95.0|99.97|88.14|90.74|85.1|80.91|
|Bulgarian|
| |bg-1.0|99.94|92.8|98.51|95.6|93.99|91.59|92.38|88.84|
| |bg-1.1|99.93|93.36|98.36|95.91|94.46|92.02|92.39|88.76|
|Buryat|
| |bxr-1.0|83.26|31.52|38.08|83.26|16.74|16.05|14.44|6.5|
|Catalan|
| |ca-1.0|99.98|99.27|98.17|98.23|96.63|97.83|92.33|89.95|
| |ca-1.1|99.99|99.51|98.2|98.22|96.72|97.8|92.14|89.6|
|Chinese|
| |zh-1.0|93.03|99.1|88.22|88.15|86.91|92.74|73.43|69.52|
| |zh-1.1|92.34|99.1|86.75|86.66|85.35|92.05|71.0|67.04|
|Croatian|
| |hr-1.0|99.92|95.56|97.66|99.92|89.49|93.85|90.61|85.77|
| |hr-1.1|99.95|95.84|97.56|99.95|89.49|94.01|89.95|84.97|
|Czech|
| |cs-1.0|99.99|83.79|98.75|95.54|93.61|95.79|90.67|88.46|
| |cs-1.1|99.99|84.19|98.54|95.33|94.09|95.7|90.72|88.52|
|Danish|
| |da-1.0|99.85|91.79|96.79|99.85|94.29|96.53|85.93|83.05|
| |da-1.1|99.82|92.64|96.52|99.82|94.39|96.21|85.09|81.83|
|Dutch|
| |nl-1.0|99.89|90.75|95.49|93.84|91.73|95.72|89.48|86.1|
| |nl-1.1|99.91|90.89|95.62|93.92|92.58|95.87|89.76|86.4|
|English|
| |en-1.0|99.25|72.8|95.34|94.83|92.48|95.62|84.7|81.93|
| |en-1.1|99.2|70.94|94.4|93.93|91.04|95.18|83.3|80.32|
|Estonian|
| |et-1.0|99.9|91.81|96.02|97.18|91.35|93.26|86.04|82.29|
| |et-1.1|99.91|91.92|96.8|97.92|93.17|93.9|86.13|82.91|
|Finnish|
| |fi-1.0|99.7|88.73|95.45|96.44|90.29|83.69|87.18|83.89|
| |fi-1.1|99.65|89.23|96.22|97.07|91.8|84.02|87.83|84.96|
|French|
| |fr-1.0|99.68|94.2|92.61|95.46|90.79|93.08|84.96|80.91|
| |fr-1.1|99.67|95.31|92.51|95.45|90.8|93.0|83.88|80.16|
|Galician|
| |gl-1.0|99.89|97.16|83.01|82.51|81.58|82.95|65.69|61.08|
| |gl-1.1|99.91|97.28|82.6|82.12|80.96|82.71|62.65|58.2|
|German|
| |de-1.0|99.7|81.19|91.38|94.26|80.37|75.8|79.6|74.35|
| |de-1.1|99.77|81.99|90.47|93.82|79.79|75.46|79.3|73.87|
|Gothic|
| |got-1.0|100.0|21.59|93.1|93.8|80.58|83.74|67.23|59.67|
|Greek|
| |el-1.0|99.88|89.46|93.7|93.54|87.14|88.92|85.63|82.05|
| |el-1.1|99.88|89.53|93.28|93.24|87.95|88.65|84.51|79.88|
|Hebrew|
| |he-1.0|99.93|99.69|54.13|54.17|51.49|54.13|34.84|32.29|
| |he-1.1|99.94|100.0|52.78|52.78|49.9|53.45|32.13|29.42|
|Hindi|
| |hi-1.0|99.98|98.84|97.16|96.43|90.29|97.48|94.66|91.26|
| |hi-1.1|100.0|99.11|96.81|96.28|89.74|97.4|94.56|90.96|
|Hungarian|
| |hu-1.0|99.8|94.18|94.52|99.8|86.22|91.07|81.57|75.95|
| |hu-1.1|99.88|97.77|93.11|99.88|86.79|91.18|77.89|70.94|
|Indonesian|
| |id-1.0|99.95|93.59|93.13|94.15|87.65|82.19|85.01|78.18|
| |id-1.1|100.0|94.58|92.95|92.81|86.27|81.51|84.73|77.99|
|Irish|
| |ga-1.0|99.56|95.38|90.95|90.07|74.1|87.51|76.32|64.74|
|Italian|
| |it-1.0|99.89|98.14|86.86|86.67|84.97|87.03|78.3|74.59|
| |it-1.1|99.92|99.07|86.58|86.4|84.53|86.75|76.38|72.35|
|Japanese|
| |ja-1.0|92.73|94.92|90.05|92.73|90.02|91.75|80.47|77.97|
| |ja-1.1|92.42|94.92|90.28|92.42|90.28|91.66|79.94|77.79|
|Kazakh|
| |kk-1.0|92.26|75.57|57.38|55.75|22.12|21.35|39.55|19.48|
|Korean|
| |ko-1.0|99.87|93.9|94.66|86.92|83.81|38.7|85.52|81.39|
| |ko-1.1|99.88|94.23|94.61|88.41|85.27|38.68|85.16|80.89|
|Kurmanji|
| |kmr-1.0|89.92|88.86|53.66|52.52|25.96|53.94|12.06|5.53|
|Latin|
| |la-1.0|99.97|92.5|97.95|93.75|91.76|96.9|89.2|86.29|
| |la-1.1|99.99|92.75|98.22|94.03|92.16|97.18|89.19|86.58|
|Latvian|
| |lv-1.0|99.66|96.35|93.43|82.52|79.99|89.47|83.04|77.98|
|North Sami|
| |sme-1.0|99.75|98.79|86.07|87.38|71.34|80.9|66.54|56.93|
|Norwegian|
| |no_bokmaal-1.0|99.92|90.93|84.24|99.92|73.68|71.68|78.24|70.83|
| |no_bokmaal-1.1|99.92|90.32|84.69|99.92|74.84|71.47|77.71|70.63|
| |no_nynorsk-1.0|99.96|91.08|97.33|99.96|93.87|85.82|90.33|88.02|
| |no_nynorsk-1.1|99.96|92.18|97.47|99.96|94.75|86.07|90.23|87.98|
|Old Church Slavonic|
| |cu-1.0|100.0|28.99|92.88|93.09|81.85|83.16|72.18|65.43|
|Persian|
| |fa-1.0|100.0|97.91|96.34|96.17|95.51|89.4|88.35|85.08|
| |fa-1.1|100.0|99.0|95.92|95.78|95.05|89.32|87.43|83.38|
|Portuguese|
| |pt-1.0|99.69|87.88|85.02|88.39|81.35|86.23|76.38|72.99|
| |pt-1.1|99.75|88.1|84.39|88.46|79.79|85.85|75.11|71.61|
|Romanian|
| |ro-1.0|99.74|95.56|97.42|96.59|95.49|96.91|90.38|85.23|
| |ro-1.1|99.71|95.42|96.96|96.32|94.98|96.57|90.14|85.06|
|Russian|
| |ru-1.0|99.71|98.79|98.4|99.71|95.55|93.89|92.7|90.97|
| |ru-1.1|99.73|98.5|98.48|99.73|95.37|93.8|92.88|90.99|
|Serbian|
| |sr-1.0|99.97|92.61|97.61|99.97|91.54|92.93|90.89|86.92|
| |sr-1.1|99.97|92.0|97.88|99.97|92.57|93.31|90.96|87.04|
|Slovak|
| |sk-1.0|99.97|86.0|95.82|82.3|78.43|90.35|88.83|85.69|
| |sk-1.1|99.95|86.67|95.33|81.01|76.98|89.87|87.64|83.84|
|Slovenian|
| |sl-1.0|99.91|97.51|97.85|92.52|91.27|96.35|91.4|89.38|
| |sl-1.1|99.87|97.64|97.62|93.29|90.99|96.36|91.46|89.19|
|Spanish|
| |es-1.0|99.98|98.32|98.0|98.0|96.62|98.05|90.53|88.27|
| |es-1.1|99.98|98.4|98.01|98.0|96.6|97.99|90.51|88.16|
|Swedish|
| |sv-1.0|99.94|92.54|97.21|95.18|92.88|97.06|88.09|84.74|
| |sv-1.1|99.36|91.22|92.74|0.0|0.0|89.37|78.14|71.86|
|Turkish|
| |tr-1.0|99.89|97.4|90.37|89.56|81.59|87.4|65.22|58.26|
| |tr-1.1|99.88|96.79|90.79|90.17|83.26|87.84|64.69|57.07|
|Ukrainian|
| |uk-1.0|99.65|93.96|96.31|88.23|86.0|92.08|86.25|82.96|
| |uk-1.1|99.76|93.58|96.0|88.17|85.39|92.28|84.9|81.04|
|Upper Sorbian|
| |hsb-1.0|98.59|69.15|59.61|98.59|37.96|22.33|11.11|3.35|
|Urdu|
| |ur-1.0|100.0|98.6|93.55|91.69|77.41|97.33|87.86|81.99|
| |ur-1.1|100.0|98.6|92.85|91.02|77.18|97.2|87.12|80.83|
|Uyghur|
| |ug-1.0|99.91|83.83|87.85|91.58|73.93|90.17|74.36|60.5|
| |ug-1.1|99.7|84.18|88.07|90.38|75.28|92.28|75.16|62.13|
|Vietnamese|
| |vi-1.0|87.2|92.88|78.35|76.43|76.18|81.47|51.59|45.49|
| |vi-1.1|86.87|92.51|76.72|74.57|72.27|81.31|50.29|43.76|
