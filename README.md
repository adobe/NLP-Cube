# NLP-Cube

NLP-Cube is an opensource Natural Language Processing Framework with support for languages which are included in the [UD Treebanks](http://universaldependencies.org/). Use NLP-Cube if you need:
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

**For user that just want to run it**, here's how to set up and use NLP-Cube in a few lines: [Quick Start Tutorial](examples/1.%20NLP-Cube%20Quick%20Tutorial.ipynb).

For **advanced users that want to create and train their own models**, please the the Advanced Tutorials in ``examples/``, starting with how to [locally install NLP-Cube](examples/2.%20Advanced%20usage%20-%20NLP-Cube%20local%20installation.ipynb).

## Simple (PIP) installation

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
cube.load("en")                 # select the desired language (it will auto-download the model)
text="This is the text I want segmented, tokenized, lemmatized and annotated with POS and dependencies."
sentences=cube(text)            # call with your own text (string) to obtain the annotations
```
The ``sentences`` object now contains the annotated text, one sentence at a time.

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

## Languages and performance


|Language|Model|Token|Sentence|UPOS|XPOS|AllTags|Lemmas|UAS|LAS|
|--------|-----|:---:|:------:|:--:|:--:|:-----:|:----:|:-:|:-:|
|Afrikaans|
| |af-1.0|99.97|99.65|97.28|93.0|91.53|96.42|87.61|83.96|
|Ancient Greek|
| |grc-1.0|100.0|18.13|94.92|95.32|84.17|86.59|72.44|67.73|
|Arabic|
| |ar-1.0|99.98|61.05|73.42|69.75|68.12|41.26|53.94|50.31|
|Basque|
| |eu-1.0|99.97|99.83|94.93|99.97|87.24|90.75|85.49|81.35|
|Bulgarian|
| |bg-1.0|99.94|92.8|98.51|95.6|93.99|91.59|92.38|88.84|
|Buryat|
| |bxr-1.0|83.26|31.52|38.08|83.26|16.74|16.05|14.44|6.5|
|Catalan|
| |ca-1.0|99.98|99.27|98.17|98.23|96.63|97.83|92.33|89.95|
|Croatian|
| |hr-1.0|99.92|95.56|97.66|99.92|89.49|93.85|90.61|85.77|
|Czech|
| |cs-1.0|99.99|83.79|98.75|95.54|93.61|95.79|90.67|88.46|
|Danish|
| |da-1.0|99.85|91.79|96.79|99.85|94.29|96.53|85.93|83.05|
|English|
| |en-1.0|99.25|72.8|95.34|94.83|92.48|95.62|84.7|81.93|
|Estonian|
| |et-1.0|99.9|91.81|96.02|97.18|91.35|93.26|86.04|82.29|
|Finnish|
| |fi-1.0|99.7|88.73|95.45|96.44|90.29|83.69|87.18|83.89|
|French|
| |fr-1.0|99.68|94.2|92.61|95.46|90.79|93.08|84.96|80.91|
|Galician|
| |gl-1.0|99.89|97.16|83.01|82.51|81.58|82.95|65.69|61.08|
|German|
| |de-1.0|99.7|81.19|91.38|94.26|80.37|75.8|79.6|74.35|
|Gothic|
| |got-1.0|100.0|21.59|93.1|93.8|80.58|83.74|67.23|59.67|
|Greek|
| |el-1.0|99.88|89.46|93.7|93.54|87.14|88.92|85.63|82.05|
|Hebrew|
| |he-1.0|99.93|99.69|54.13|54.17|51.49|54.13|34.84|32.29|
|Hindi|
| |hi-1.0|99.98|98.84|97.16|96.43|90.29|97.48|94.66|91.26|
|Hungarian|
| |hu-1.0|99.8|94.18|94.52|99.8|86.22|91.07|81.57|75.95|
|Irish|
| |ga-1.0|99.56|95.38|90.95|90.07|74.1|87.51|76.32|64.74|
|Old Church Slavonic|
| |cu-1.0|100.0|28.99|92.88|93.09|81.85|83.16|72.18|65.43|
|Persian|
| |fa-1.0|100.0|97.91|96.34|96.17|95.51|89.4|88.35|85.08|
|Spanish|
| |es-1.0|99.98|98.32|98.0|98.0|96.62|98.05|90.53|88.27|
|Upper Sorbian|
| |hsb-1.0|98.59|69.15|59.61|98.59|37.96|22.33|11.11|3.35|


Results are reported against the test files for each language (available in the UD 2.2 corpus) using the 2018 conll eval script.

Note: we are in the process of training version 1.1 of the models which do not require embeddings (performance will be roughly similar, but models won't need the large vector embedding files).
```
