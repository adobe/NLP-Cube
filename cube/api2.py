# -*- coding: utf-8 -*-
import logging
import sys
import os
"""
Usage:
import cube
nlp = cube.load("en")
"""
from cube.io_utils.modelstore import ModelStore
from cube.io_utils.objects import Doc, Sentence, Word, Token
from typing import Optional

def load(lang:str, log_level:str = "ERROR", use_gpu=True) :
    # TODO set log level

    # gets components (by download or use from local cache)
    paths = ModelStore.solve(lang)

    # instantiate cube object and load components
    cube = Cube(component_paths=paths)

    return cube

class Cube(object):
    def __init__(self, component_paths:dict):
        self.lang_id = 0
        self.tokenizer = None
        self.compound = None
        self.lemmatizer = None
        self.tagger_UPOS = None
        self.tagger_XPOS = None
        self.tagger_ATTRS = None
        self.parser = None

        if "tokenizer" in component_paths:
            from cube.networks.tokenizer import Tokenizer
            from cube.io_utils.encodings import Encodings
            from cube.io_utils.config import TokenizerConfig
            tokenizer_config = TokenizerConfig(filename=component_paths["tokenizer"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["tokenizer"]["encodings"])
            self.tokenizer = Tokenizer(config=tokenizer_config, encodings=encodings)
            self.tokenizer.load(path=component_paths["tokenizer"]["model"])

        if "compound" in component_paths:
            from cube.networks.compound import Compound
            from cube.io_utils.encodings import Encodings
            from cube.io_utils.config import CompoundConfig
            compound_config = CompoundConfig(filename=component_paths["compound"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["compound"]["encodings"])
            self.compound = Compound(config=compound_config, encodings=encodings)
            self.compound.load(path=component_paths["compound"]["model"])

        if "lemmatizer" in component_paths:
            from cube.io_utils.config import LemmatizerConfig
            from cube.io_utils.encodings import Encodings
            from cube.networks.lemmatizer import Lemmatizer
            lemmatizer_config = LemmatizerConfig(filename=component_paths["lemmatizer"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["lemmatizer"]["encodings"])
            self.lemmatizer = Lemmatizer(config=lemmatizer_config, encodings=encodings)
            self.lemmatizer.load(path=component_paths["lemmatizer"]["model"])

        if "tagger" in component_paths:
            from cube.io_utils.config import TaggerConfig
            from cube.io_utils.encodings import Encodings
            from cube.networks.tagger import Tagger
            tagger_config = TaggerConfig(filename=component_paths["tagger"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["tagger"]["encodings"])
            self.tagger_UPOS = Tagger(config=tagger_config, encodings=encodings)
            self.tagger_UPOS.load(path=component_paths["tagger"]["model_UPOS"])
            self.tagger_XPOS = Tagger(config=tagger_config, encodings=encodings)
            self.tagger_XPOS.load(path=component_paths["tagger"]["model_XPOS"])
            self.tagger_ATTRS = Tagger(config=tagger_config, encodings=encodings)
            self.tagger_ATTRS.load(path=component_paths["tagger"]["model_ATTRS"])

        if "parser" in component_paths:
            from cube.io_utils.config import ParserConfig
            from cube.io_utils.encodings import Encodings
            from cube.networks.parser import Parser
            parser_config = ParserConfig(filename=component_paths["parser"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["parser"]["encodings"])
            self.parser = Parser(config=parser_config, encodings=encodings)
            self.parser.load(path=component_paths["parser"]["model"])

    def __call__(self, text):
        sequences = []
        if self.tokenizer:
            if not isinstance(text, str):
                raise Exception("The text argument must be a string!")
            # split text by lines
            input_lines = text.split("\n")
            for input_line in input_lines:                
                sequences+=self.tokenizer.process(input_line, self.lang_id)
        else:
            if not isinstance(text, list):
                raise Exception("The text argument must be a list of lists of tokens!")
            sequences = text  # the input should already be tokenized

        if self.compound:
            sequences = self.compound.process(sequences, self.lang_id)

        if self.parser:
            sequences = self.parser.process(sequences, self.lang_id)

        if self.tagger_UPOS or self.lemmatizer:
            sequences = self.tagger_UPOS.process(sequences, upos=True, lang_id = self.lang_id)
            sequences = self.tagger_XPOS.process(sequences, xpos=True, lang_id = self.lang_id)
            sequences = self.tagger_ATTRS.process(sequences, attrs=True, lang_id = self.lang_id)

        #from ipdb import set_trace
        #set_trace()

        for seq in sequences:
            for entry in seq:
                sys.stdout.write(str(entry))
            print("")

        if self.lemmatizer:
            sequences = self.lemmatizer.process(sequences, self.lang_id)

        for seq in sequences:
            for entry in seq:
                sys.stdout.write(str(entry))
            print("")

        return sequences


if __name__ == "__main__":
   nlp = load("hy")

   #r = nlp("This is a test.")
   r = nlp("Պարտությունը չունի բարոյականություն և չունի հայրենիք, ինչպես` քաոսը2")