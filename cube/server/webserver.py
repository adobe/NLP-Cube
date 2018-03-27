#
# Author: Tiberiu Boros
#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from flask import Flask
from flask import Response
import sys

app = Flask(__name__)
singletonServer = False


@app.route('/')
def index():
    return "NLP server running"


@app.route('/test')
def test():
    global singletonServer
    seqs = singletonServer.tokenizer.tokenize("Ana are mere dar nu are pere. ")
    result = ""
    for seq in seqs:
        if len(seq) > 0:
            tagged = singletonServer.parser.tag(seq)
            iOrig = 0
            iTags = 0
            while iOrig < len(seq):
                while seq[iOrig].is_compound_entry:
                    iOrig += 1
                seq[iOrig].upos = tagged[iTags].upos
                seq[iOrig].xpos = tagged[iTags].xpos
                seq[iOrig].attrs = tagged[iTags].attrs
                seq[iOrig].head = tagged[iTags].head
                seq[iOrig].label = tagged[iTags].label
                iTags += 1
                iOrig += 1

            for entry in seq:
                result += str(entry.index) + "\t" + str(entry.word) + "\t" + str(entry.lemma) + "\t" + str(
                    entry.upos) + "\t" + str(entry.xpos) + "\t" + str(entry.attrs) + "\t" + str(
                    entry.head) + "\t" + str(entry.label) + "\t" + str(entry.deps) + "\t" + entry.space_after + "\n"
            result += "\n"

    return Response(result, mimetype='text/plain')


@app.route('/nlp', methods=['GET', 'POST'])
def nlp():
    query = ""
    from flask import request
    if request.args.get('q'):
        query = request.args.get('q')
    else:
        query = request.form.get('q')
    # from ipdb import set_trace
    # set_trace()
    query += u' '
    # if
    query = query.encode('utf-8')
    global singletonServer
    seqs = singletonServer.tokenizer.tokenize(query)
    result = ""
    for seq in seqs:
        if len(seq) > 0:
            tagged = singletonServer.parser.tag(seq)

            iOrig = 0
            iTags = 0
            while iOrig < len(seq):
                while seq[iOrig].is_compound_entry:
                    iOrig += 1
                seq[iOrig].upos = tagged[iTags].upos
                seq[iOrig].xpos = tagged[iTags].xpos
                seq[iOrig].attrs = tagged[iTags].attrs
                seq[iOrig].head = tagged[iTags].head
                seq[iOrig].label = tagged[iTags].label

                iTags += 1
                iOrig += 1
            if singletonServer.lemmatizer is not None:
                lemmatized = singletonServer.lemmatizer.tag(seq)
            else:
                lemmatized = [entry.word.decode('utf-8') for entry in seq]

            for entry, lemma in zip(seq, lemmatized):
                if not entry.is_compound_entry:
                    entry.lemma = lemma.encode('utf-8')

            for entry in seq:
                if entry.lemma is None:
                    entry.lemma = ""
                result += str(entry.index) + "\t" + str(entry.word) + "\t" + str(
                    entry.lemma) + "\t" + str(
                    entry.upos) + "\t" + str(entry.xpos) + "\t" + str(entry.attrs) + "\t" + str(
                    entry.head) + "\t" + str(entry.label) + "\t" + str(entry.deps) + "\t" + entry.space_after + "\n"
            result += "\n"

    return Response(result, mimetype='text/plain')


class EmbeddedWebserver:
    def __init__(self, embeddings, port=80, tokenization=None, lemma=None, tagging=None, parsing=None):
        global singletonServer
        singletonServer = self
        if tokenization is not None:
            sys.stdout.write("Loading tokenization model from " + tokenization)
            sys.stdout.flush()
            from generic_networks.tokenizers import TieredTokenizer
            from io_utils.config import TieredTokenizerConfig
            from io_utils.encodings import Encodings
            tok_encodings = Encodings()
            tok_encodings.load(tokenization + ".encodings")
            tok_config = TieredTokenizerConfig()
            tok_config.load(tokenization + ".conf")
            self.tokenizer = TieredTokenizer(tok_config, tok_encodings, embeddings, runtime=True)
            self.tokenizer.load(tokenization)

        if parsing is not None:
            sys.stdout.write("Loading parsing model from " + parsing)
            from generic_networks.parsers import BDRNNParser
            from io_utils.config import ParserConfig
            from io_utils.encodings import Encodings
            parse_encodings = Encodings()
            parse_encodings.load(parsing + ".encodings")
            parse_config = ParserConfig()
            parse_config.load(parsing + ".conf")
            self.parser = BDRNNParser(parse_config, parse_encodings, embeddings, runtime=True)
            self.parser.load(parsing + ".bestUAS")

        if lemma is not None:
            sys.stdout.write("Loading lemma model from " + lemma)
            from generic_networks.lemmatizers import FSTLemmatizer
            from io_utils.config import LemmatizerConfig
            from io_utils.encodings import Encodings
            lemma_encodings = Encodings()
            lemma_encodings.load(lemma + ".encodings")
            lemma_config = LemmatizerConfig()
            lemma_config.load(lemma + ".conf")
            self.lemmatizer = FSTLemmatizer(lemma_config, lemma_encodings, embeddings, runtime=True)
            self.lemmatizer.load(lemma + ".bestACC")
        else:
            self.lemmatizer = None

        global app
        app.run(port=port)
        self.port = port
