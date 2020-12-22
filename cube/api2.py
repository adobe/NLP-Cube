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

def load(lang:str,  log_level:str = "ERROR", use_gpu=True) :
    # TODO set log level

    # gets components (by download or use from local cache)
    paths = ModelStore.solve(lang)

    # instantiate cube object and load components
    cube = Cube(component_paths=paths)

    return cube

class Cube(object):
    def __init__(self, component_paths:dict):
        self.tokenizer = None
        self.lemmatizer = None
        self.tagger_UPOS = None
        self.tagger_XPOS = None
        self.tagger_ATTRS = None
        self.parser_UAS = None
        self.parser_LAS = None

        if "tokenizer" in component_paths:
            from cube.networks.tokenizer import Tokenizer
            from cube.io_utils.encodings import Encodings
            from cube.config import TokenizerConfig
            tokenizer_config = TokenizerConfig(filename=component_paths["tokenizer"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["tokenizer"]["encodings"])
            self.tokenizer = Tokenizer(config=tokenizer_config, encodings=encodings)
            self.tokenizer.load(path=component_paths["tokenizer"]["model"])

        if "lemmatizer" in component_paths:
            from cube.config import LemmatizerConfig
            from cube.io_utils.encodings import Encodings
            from cube.networks.lemmatizer import Lemmatizer
            lemmatizer_config = LemmatizerConfig(filename=component_paths["lemmatizer"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["lemmatizer"]["encodings"])
            self.lemmatizer = Lemmatizer(config=lemmatizer_config, encodings=encodings)
            self.lemmatizer.load(path=component_paths["lemmatizer"]["model"])

        if "tagger" in component_paths:
            from cube.config import TaggerConfig
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
            from cube.config import ParserConfig
            from cube.io_utils.encodings import Encodings
            from cube.networks.parser import Parser
            parser_config = ParserConfig(filename=component_paths["parser"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["parser"]["encodings"])
            self.parser_UAS = Parser(config=parser_config, encodings=encodings)
            self.parser_UAS.load(path=component_paths["parser"]["model_UAS"])
            self.parser_LAS = Parser(config=parser_config, encodings=encodings)
            self.parser_LAS.load(path=component_paths["parser"]["model_LAS"])


    def __call__(self, text):

        sequences = []
        if self._tokenizer:
            if not isinstance(text, str):
                raise Exception("The text argument must be a string!")
            # split text by lines
            input_lines = text.split("\n")
            for input_line in input_lines:                
                sequences+=self.tokenizer(input_line)
        else:
            if not isinstance(text, list):
                raise Exception("The text argument must be a list of lists of tokens!")
            sequences = text  # the input should already be tokenized

        for seq in sequences:
            for entry in seq:
                sys.stdout.write(str(entry))
            print("")

        #if self._compound_word_expander:
        #    sequences = self._compound_word_expander.expand_sequences(sequences)

        #if self.parser_UAS:
        #    sequences = self.parser.parse_sequences(sequences)

        if self._tagger or self._lemmatizer:
            import copy
            new_sequences = []
            for sequence in sequences:                
                new_sequence = copy.deepcopy(sequence)                
                predicted_tags_UPOS = self._tagger[0].tag(new_sequence)
                predicted_tags_XPOS = self._tagger[1].tag(new_sequence)
                predicted_tags_ATTRS = self._tagger[2].tag(new_sequence)
                for entryIndex in range(len(new_sequence)):
                    new_sequence[entryIndex].upos = predicted_tags_UPOS[entryIndex][0]
                    new_sequence[entryIndex].xpos = predicted_tags_XPOS[entryIndex][1]
                    new_sequence[entryIndex].attrs = predicted_tags_ATTRS[entryIndex][2]
                new_sequences.append(new_sequence)
            sequences = new_sequences

        if self._lemmatizer:
            sequences = self._lemmatizer.lemmatize_sequences(sequences)

        return sequences


if __name__ == "__main__":
   pass