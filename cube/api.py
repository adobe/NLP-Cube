# -*- coding: utf-8 -*-
import logging
import sys
import os
sys.path.append('')
from cube.networks.lm import LMHelperHF
from cube.networks.utils import MorphoCollate, Word2TargetCollate
from cube.networks.utils_tokenizer import TokenCollateHF

sys.path.append('')
"""
Usage:
import cube
nlp = cube.load("en")
"""
from cube.io_utils.modelstore import ModelStore
#from cube.io_utils.objects import Doc, Sentence, Word, Token
from typing import Optional


def load(lang: str, log_level: str = "ERROR", device="cpu", batch_size:int=1, num_workers:int=0, verbose=False):
    # TODO set log level

    # gets component paths (by downloading them or from the local cache)
    component_paths, lang_id = ModelStore.solve(lang)

    # instantiate cube object and load components
    cube = Cube(component_paths=component_paths, device=device, batch_size=batch_size, model_name=lang, verbose=verbose)
    cube.lang_id = lang_id

    return cube


class Cube(object):
    def __init__(self, component_paths: dict, device="cpu", batch_size:int=1, num_workers:int=0, model_name:str="", verbose:bool=False):
        self.lang_id = 0
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name=model_name
        self.verbose=verbose

        self.tokenizer = None
        self.compound = None
        self.lemmatizer = None
        self.tagger_UPOS = None
        self.tagger_XPOS = None
        self.tagger_ATTRS = None
        self.parser = None

        print(component_paths)

        # load LM Helper TODO: allow other helpers, allow other languages (future version)
        self.lmhelper = LMHelperHF(model='xlm-roberta-base', device=device)
        if verbose:
            print(f"\t loaded language model helper")


        if "tokenizer" in component_paths and component_paths["tokenizer"]:
            from cube.networks.tokenizer import Tokenizer
            from cube.io_utils.encodings import Encodings
            from cube.io_utils.config import TokenizerConfig
            tokenizer_config = TokenizerConfig(filename=component_paths["tokenizer"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["tokenizer"]["encodings"])
            self.tokenizer = Tokenizer(config=tokenizer_config, encodings=encodings, ext_word_emb=self.lmhelper.get_embedding_size())
            self.tokenizer.load(model_path=component_paths["tokenizer"]["model"])

            self.tokenizer_collate_fn = TokenCollateHF(encodings, lm_model='xlm-roberta-base', lm_device=self.device) # TODO lm_model fixed
            if verbose:
                print(f"\t loaded tokenizer model : {component_paths['tokenizer']['model']}")

        if "cwe" in component_paths and component_paths["cwe"]:
            from cube.networks.compound import Compound
            from cube.io_utils.encodings import Encodings
            from cube.io_utils.config import CompoundConfig
            compound_config = CompoundConfig(filename=component_paths["cwe"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["cwe"]["encodings"])
            self.compound = Compound(config=compound_config, encodings=encodings)
            self.compound.load(model_path=component_paths["cwe"]["model"])
            self.compound_collate_fn = Word2TargetCollate(encodings)
            if verbose:
                print(f"\t loaded compound word expander model : {component_paths['cwe']['model']}")

        if "lemmatizer" in component_paths and component_paths["lemmatizer"]:
            from cube.io_utils.config import LemmatizerConfig
            from cube.io_utils.encodings import Encodings
            from cube.networks.lemmatizer import Lemmatizer
            lemmatizer_config = LemmatizerConfig(filename=component_paths["lemmatizer"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["lemmatizer"]["encodings"])
            self.lemmatizer = Lemmatizer(config=lemmatizer_config, encodings=encodings)
            self.lemmatizer.load(model_path=component_paths["lemmatizer"]["model"])
            self.lemmatizer_collate_fn = Word2TargetCollate(encodings)
            if verbose:
                print(f"\t loaded lemmatizer model : {component_paths['lemmatizer']['model']}")

        if "tagger" in component_paths and component_paths["tagger"]:
            from cube.io_utils.config import TaggerConfig
            from cube.io_utils.encodings import Encodings
            from cube.networks.tagger import Tagger
            tagger_config = TaggerConfig(filename=component_paths["tagger"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["tagger"]["encodings"])
            self.tagger_UPOS = Tagger(config=tagger_config, encodings=encodings)
            self.tagger_UPOS.load(model_path=component_paths["tagger"]["model_UPOS"])
            self.tagger_XPOS = Tagger(config=tagger_config, encodings=encodings)
            self.tagger_XPOS.load(model_path=component_paths["tagger"]["model_XPOS"])
            self.tagger_ATTRS = Tagger(config=tagger_config, encodings=encodings)
            self.tagger_ATTRS.load(model_path=component_paths["tagger"]["model_ATTRS"])

        if "parser" in component_paths and component_paths["parser"]:
            from cube.io_utils.config import ParserConfig
            from cube.io_utils.encodings import Encodings
            from cube.networks.parser import Parser
            parser_config = ParserConfig(filename=component_paths["parser"]["config"])
            encodings = Encodings()
            encodings.load(filename=component_paths["parser"]["encodings"])
            self.parser = Parser(config=parser_config, encodings=encodings, ext_word_emb=self.lmhelper.get_embedding_size())
            self.parser.load(model_path=component_paths["parser"]["model"])
            self.parser_collate_fn = MorphoCollate(encodings)
            if verbose:
                print(f"\t loaded parser model : {component_paths['parser']['model']}")

        print("Model loaded!")

    def __call__(self, data):
        if isinstance(data, str):
            if self.tokenizer is None:
                raise Exception("Tokenizer needs raw text as input!")

            doc = self.tokenizer.process(data, self.tokenizer_collate_fn, lang_id=self.lang_id, batch_size=self.batch_size)
        else:
            doc = data # TODO: check data is a Document object and check validity

            print("After tokenizer")
            print(doc)

        print("Applying LMHelper on document")
        self.lmhelper.apply(doc)

        # apply lang_id on each sentence, for the parser
        for ii in range(len(doc.sentences)):
            doc.sentences[ii].lang_id = self.lang_id

        if self.compound:
            doc = self.compound.process(doc, collate=self.compound_collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)

            print("After compound")
            print(doc)


        if self.parser:
            doc = self.parser.process(doc, collate=self.parser_collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)

            print("After parser")
            print(doc)


        if self.lemmatizer:
            doc = self.lemmatizer.process(doc, collate=self.lemmatizer_collate_fn, batch_size=self.batch_size, num_workers=self.num_workers)

            print("After lemmatizer")
            print(doc)

        return doc


if __name__ == "__main__":
    import cube
    nlp = cube.load("es_gsd", device="cpu", verbose=True)

    #doc = nlp(
    #    "Acesta este un simplu test. Ana are mere dar nu are pere și mănâncă banane.\nHai să vedem ce face când dă de ENTER. Știu că avea și o problemă cu băiatul, băieții sau băieților, din cauza corpusului de antrenare.")
    #doc = nlp("Trees have apples. Another sentence.")
    #doc = nlp("עשרות אנשים מגיעים מתאילנד לישראל כשהם נרשמים כמתנדבים, אך למעשה משמשים עובדים שכירים זולים.")
    doc = nlp("Vámonos al mar.")

    print(doc)

