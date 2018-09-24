# -*- coding: utf-8 -*-

import sys
import os
from .io_utils.encodings import Encodings
from .io_utils.embeddings import WordEmbeddings
from .io_utils.model_store import ModelMetadata, ModelStore
from .io_utils.config import TieredTokenizerConfig, CompoundWordConfig, LemmatizerConfig, TaggerConfig, ParserConfig
from .generic_networks.tokenizers import TieredTokenizer
from .generic_networks.token_expanders import CompoundWordExpander
from .generic_networks.lemmatizers import FSTLemmatizer
from .generic_networks.taggers import BDRNNTagger
from .generic_networks.parsers import BDRNNParser

from pathlib import Path



class Cube(object):
    def __init__(self, verbose=False):
        """
        Create an empty instance for Cube
        Before it can be used, you must call @method load with @param language_code set to your target language
        """
        self._loaded = False
        self._verbose = verbose
        self._tokenizer = None  # tokenizer object, default is None
        self._compound_word_expander = False  # compound word expander, default is None
        self._lemmatizer = False  # lemmatizer object, default is None
        self._parser = False  # parser object, default is None
        self._tagger = False  # tagger object, default is None
        self.embeddings = None  # ?? needed?
        self.metadata = ModelMetadata()
        self._model_repository = os.path.join(str(Path.home()), ".nlpcube/models")
        if not os.path.exists(self._model_repository):
            os.makedirs(self._model_repository)

        self._embeddings_repository = os.path.join(self._model_repository, "embeddings")

    def load(self, language_code, version="latest", tokenization=True, compound_word_expanding=False, tagging=True,
             lemmatization=True, parsing=True):
        """
        Loads the pipeline with all available models for the target language.

        @param lang_code: Target language code. See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
        @param version: "latest" to get the latest version, or other specific version in like "1.0", "2.1", etc .
       
        """
        # Initialize a ModelStore object
        model_store_object = ModelStore(disk_path=self._model_repository)

        # Find a local model or download it if it does not exist, returning the local model folder path
        model_folder_path = model_store_object.find(lang_code=language_code, version=version, verbose=self._verbose)

        # Load metadata from the model 
        self.metadata.read(os.path.join(model_folder_path, "metadata.json"))

        # Load embeddings                
        embeddings = WordEmbeddings(verbose=False)
        if self._verbose:
            sys.stdout.write('\tLoading embeddings... \n')
        embeddings.read_from_file(os.path.join(self._embeddings_repository, self.metadata.embeddings_file_name), None,
                                  full_load=False)

        # 1. Load tokenizer
        if tokenization:
            if not os.path.isfile(os.path.join(model_folder_path, 'tokenizer-tok.bestAcc')):
                sys.stdout.write('\tTokenization is not available on this model. \n')
            else:
                if self._verbose:
                    sys.stdout.write('\tLoading tokenization model ...\n')
                tokenizer_encodings = Encodings(verbose=False)
                tokenizer_encodings.load(os.path.join(model_folder_path, 'tokenizer.encodings'))
                config = TieredTokenizerConfig(os.path.join(model_folder_path, 'tokenizer.conf'))
                self._tokenizer = TieredTokenizer(config, tokenizer_encodings, embeddings, runtime=True)
                self._tokenizer.load(os.path.join(model_folder_path, 'tokenizer'))

                # 3. Load compound
        if compound_word_expanding:
            if not os.path.isfile(os.path.join(model_folder_path, 'compound.bestAcc')):
                if self._verbose:  # supress warning here because many languages do not have compund words
                    sys.stdout.write('\tCompound word expansion is not available on this model. \n')
            else:
                if self._verbose:
                    sys.stdout.write('\tLoading compound word expander model ...\n')
                compound_encodings = Encodings(verbose=False)
                compound_encodings.load(os.path.join(model_folder_path, 'compound.encodings'))
                config = CompoundWordConfig(os.path.join(model_folder_path, 'compound.conf'))
                self._compound_word_expander = CompoundWordExpander(config, compound_encodings, embeddings,
                                                                    runtime=True)
                self._compound_word_expander.load(os.path.join(model_folder_path, 'compound.bestAcc'))

        # 4. Load lemmatizer
        if lemmatization:
            if not os.path.isfile(os.path.join(model_folder_path, 'lemmatizer.bestAcc')):
                sys.stdout.write('\tLemmatizer is not available on this model. \n')
            else:
                if self._verbose:
                    sys.stdout.write('\tLoading lemmatization model ...\n')
                lemmatizer_encodings = Encodings(verbose=False)
                lemmatizer_encodings.load(os.path.join(model_folder_path, 'lemmatizer.encodings'))
                config = LemmatizerConfig(os.path.join(model_folder_path, 'lemmatizer.conf'))
                self._lemmatizer = FSTLemmatizer(config, lemmatizer_encodings, embeddings, runtime=True)
                self._lemmatizer.load(os.path.join(model_folder_path, 'lemmatizer.bestAcc'))

                # 5. Load taggers
        if tagging or lemmatization:  # we need tagging for lemmatization
            if not os.path.isfile(os.path.join(model_folder_path, 'tagger.bestUPOS')):
                sys.stdout.write('\tTagging is not available on this model. \n')
                if lemmatization:
                    sys.stdout.write('\t\tDisabling the lemmatization model due to missing tagger. \n')
                    self._lemmatizer = None
            else:
                if self._verbose:
                    sys.stdout.write('\tLoading tagger model ...\n')
                tagger_encodings = Encodings(verbose=False)
                tagger_encodings.load(os.path.join(model_folder_path, 'tagger.encodings'))
                config = TaggerConfig(os.path.join(model_folder_path, 'tagger.conf'))
                self._tagger = [None, None, None]
                self._tagger[0] = BDRNNTagger(config, tagger_encodings, embeddings, runtime=True)
                self._tagger[0].load(os.path.join(model_folder_path, 'tagger.bestUPOS'))
                self._tagger[1] = BDRNNTagger(config, tagger_encodings, embeddings, runtime=True)
                self._tagger[1].load(os.path.join(model_folder_path, 'tagger.bestXPOS'))
                self._tagger[2] = BDRNNTagger(config, tagger_encodings, embeddings, runtime=True)
                self._tagger[2].load(os.path.join(model_folder_path, 'tagger.bestATTRS'))

        # 6. Load parser
        if parsing:
            if not os.path.isfile(os.path.join(model_folder_path, 'parser.bestUAS')):
                sys.stdout.write('\tParsing is not available on this model... \n')
            else:
                if self._verbose:
                    sys.stdout.write('\tLoading parser model ...\n')
                parser_encodings = Encodings(verbose=False)
                parser_encodings.load(os.path.join(model_folder_path, 'parser.encodings'))
                config = ParserConfig(os.path.join(model_folder_path, 'parser.conf'))
                self._parser = BDRNNParser(config, parser_encodings, embeddings, runtime=True)
                self._parser.load(os.path.join(model_folder_path, 'parser.bestUAS'))

        self._loaded = True
        if self._verbose:
            sys.stdout.write('Model loading complete.\n\n')

    def __call__(self, text):
        if not self._loaded:
            raise Exception("Cube object is initialized but no model is loaded (eg.: call cube.load('en') )")

        sequences = []
        if self._tokenizer:
            # split text by lines
            input_lines = text.split("\n")
            for input_line in input_lines:
                sequences += self._tokenizer.tokenize(input_line)
        else:
            sequences = text  # the input should already be tokenized


        if self._compound_word_expander:
            sequences = self._compound_word_expander.expand_sequences(sequences)

        if self._parser:
            sequences = self._parser.parse_sequences(sequences)

        if self._tagger or self._lemmatizer:
            import copy
            new_sequences = []
            for sequence in sequences:
                new_sequence = copy.deepcopy(sequence)
                predicted_tags_UPOS = self._tagger[0].tag(new_sequence)
                predicted_tags_XPOS = self._tagger[1].tag(new_sequence)
                predicted_tags_ATTRS = self._tagger[2].tag(new_sequence)
                for entryIndex in range(len(sequence)):
                    new_sequence[entryIndex].upos = predicted_tags_UPOS[entryIndex][0]
                    new_sequence[entryIndex].xpos = predicted_tags_XPOS[entryIndex][1]
                    new_sequence[entryIndex].attrs = predicted_tags_ATTRS[entryIndex][2]
                new_sequences.append(new_sequence)
            sequences = new_sequences

        if self._lemmatizer:
            sequences = self._lemmatizer.lemmatize_sequences(sequences)

        return sequences


if __name__ == "__main__":
    cube = Cube(verbose=True)
    cube.load('en', tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True)
    cube.metadata.info()

    text = "Je suis un cochon. Prințesa Louisa s-a născut la 19 martie la Casa Leicester, Westminster, Londra. Tatăl ei a fost Frederick, Prinț de Wales, fiul cel mare al regelui George al II-lea și a reginei Caroline de Ansbach. Mama ei a fost Prințesa de Wales (născută Augusta de Saxa-Gotha).\n\n\rA fost botezată la 11 aprilie și nașii ei au fost: Frederic al II-lea, Landgraf de Hesse-Cassel (unchiul patern prin căsătorie) și mătușile paterne Louise, regină a Danemarcei și Norvegiei și Anne, Prințesă Regală.[1]\n\rSănătatea ei a fost delicată de-a lungul întregii ei vieți. Prințesa Louisa a murit la Casa Carlton din Londra, la 13 mai 1768, necăsătorită și fără copii, la vârsta de 19 ani."

    sentences = cube(text)

    for sentence in sentences:
        print()
        for token in sentence:
            line = ""
            line += str(token.index) + "\t"
            line += token.word + "\t"
            line += token.lemma + "\t"
            line += token.upos + "\t"
            line += token.xpos + "\t"
            line += token.attrs + "\t"
            line += str(token.head) + "\t"
            line += token.label + "\t"
            line += token.deps + "\t"
            line += token.space_after
            print(line)
