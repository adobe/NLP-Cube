# -*- coding: utf-8 -*-

import sys
import os
from io_utils.encodings import Encodings
from io_utils.embeddings import WordEmbeddings

BASE_PATH = '/opt/cube_models'


class PipelineComponents:
    TOKENIZER = 'tokenizer'
    COMPOUND = 'compound'
    TAGGER = 'tagger'
    PARSER = 'parser'
    LEMMATIZER = 'lemmatizer'


class Cube:
    def __init__(self):
        """
        Create an empty instance for Cube
        Before it can be used, you must call @method load with @param language_code set to your target language
        """
        self.loaded = False
        self.tokenizer_enabled = False
        self.compound_enabled = False
        self.lemmatizer_enabled = False
        self.parser_enabled = False
        self.tokenizer_enabled = False
        self.tagger_enabled = False
        self.models = {}
        self.embeddings = None

    def download_models(self, lang_code):
        """
        Downloads pre-trained models for the desired language. All existing models will be overwritten
        @param lang_code: Target language code. See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
        @return: True if the download was successful, False otherwise
        """
        sys.stdout.write('TODO: Downloading models for ' + lang_code + "\n")

    def load(self, lang_code, base_path=None):
        """
        Loads the pipeline with all available models for the target language
        @param lang_code: Target language code. See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
        @param base_path: Base path for models. Only required for custom-trained models. Otherwise, just leave this parameter untouched to use the default model location
        @return: True if loading was successful, False otherwise
        """
        sys.stdout.write('Loading models for ' + lang_code + "\n")
        if base_path is None:
            global BASE_PATH
            base_path = BASE_PATH

        self.embeddings = WordEmbeddings()
        self.embeddings.read_from_file(os.path.join(base_path, lang_code + "/wiki." + lang_code + ".vec"), None,
                                       full_load=False)
        if not os.path.isfile(os.path.join(base_path, lang_code + "/tokenizer-tok.bestAcc")):
            sys.stdout.write(
                "\tTokenization disabled. \n")
        else:
            self.tokenizer_enabled = True
            sys.stdout.write("\tTokenization enabled.\n")
            tokenizer_encodings = Encodings(verbose=False)
            tokenizer_encodings.load(os.path.join(base_path, lang_code + "/tokenizer.encodings"))
            from io_utils.config import TieredTokenizerConfig
            from generic_networks.tokenizers import TieredTokenizer
            config = TieredTokenizerConfig(os.path.join(base_path, lang_code + "/tokenizer.conf"))
            tokenizer_object = TieredTokenizer(config, tokenizer_encodings, self.embeddings, runtime=True)
            tokenizer_object.load(os.path.join(base_path, lang_code + "/tokenizer"))
            self.models[PipelineComponents.TOKENIZER] = tokenizer_object

        if not os.path.isfile(os.path.join(base_path, lang_code + "/compound.bestAcc")):
            sys.stdout.write(
                "\tCompound disabled. \n")
        else:
            self.compound_enabled = True
            sys.stdout.write("\tCompound enabled.\n")
            compound_encodings = Encodings(verbose=False)
            compound_encodings.load(os.path.join(base_path, lang_code + "/compound.encodings"))
            from io_utils.config import CompoundWordConfig
            from generic_networks.token_expanders import CompoundWordExpander
            config = CompoundWordConfig(os.path.join(base_path, lang_code + "/compound.conf"))
            compound_object = CompoundWordExpander(config, compound_encodings, self.embeddings, runtime=True)
            compound_object.load(os.path.join(base_path, lang_code + "/compound.bestAcc"))
            self.models[PipelineComponents.COMPOUND] = compound_object

        if not os.path.isfile(os.path.join(base_path, lang_code + "/lemmatizer.bestACC")):
            sys.stdout.write(
                "\tLemmatizer disabled. \n")
        else:
            self.lemmatizer_enabled = True
            sys.stdout.write("\tLemmatizer enabled.\n")
            lemmatizer_encodings = Encodings(verbose=False)
            lemmatizer_encodings.load(os.path.join(base_path, lang_code + "/lemmatizer.encodings"))
            from io_utils.config import LemmatizerConfig
            from generic_networks.lemmatizers import FSTLemmatizer
            config = LemmatizerConfig(os.path.join(base_path, lang_code + "/lemmatizer.conf"))
            lemmatizer_object = FSTLemmatizer(config, lemmatizer_encodings, self.embeddings, runtime=True)
            lemmatizer_object.load(os.path.join(base_path, lang_code + "/lemmatizer.bestACC"))
            self.models[PipelineComponents.LEMMATIZER] = lemmatizer_object

        if not os.path.isfile(os.path.join(base_path, lang_code + "/tagger.bestUPOS")):
            sys.stdout.write(
                "\tTagger disabled. \n")
        else:
            self.tagger_enabled = True
            sys.stdout.write("\tTagger enabled.\n")
            tagger_encodings = Encodings(verbose=False)
            tagger_encodings.load(os.path.join(base_path, lang_code + "/tagger.encodings"))
            from io_utils.config import TaggerConfig
            from generic_networks.taggers import BDRNNTagger
            config = TaggerConfig(os.path.join(base_path, lang_code + "/tagger.conf"))

            tagger_upos_object = BDRNNTagger(config, tagger_encodings, self.embeddings, runtime=True)
            tagger_upos_object.load(os.path.join(base_path, lang_code + "/tagger.bestUPOS"))
            tagger_xpos_object = BDRNNTagger(config, tagger_encodings, self.embeddings, runtime=True)
            tagger_xpos_object.load(os.path.join(base_path, tagger_encodings + "/tagger.bestXPOS"))
            tagger_attrs_object = BDRNNTagger(config, tagger_encodings, self.embeddings, runtime=True)
            tagger_attrs_object.load(os.path.join(base_path, lang_code + "/tagger.bestATTRS"))

            self.models[PipelineComponents.TAGGER] = [tagger_upos_object, tagger_xpos_object, tagger_attrs_object]

        if not os.path.isfile(os.path.join(base_path, lang_code + "/parser.bestUAS")):
            sys.stdout.write(
                "\tParser disabled. \n")
        else:
            self.parser_enabled = True
            sys.stdout.write("\tParser enabled.\n")
            lemmatizer_encodings = Encodings(verbose=False)
            lemmatizer_encodings.load(os.path.join(base_path, lang_code + "/parser.encodings"))
            from io_utils.config import ParserConfig
            from generic_networks.parsers import BDRNNParser
            config = ParserConfig(os.path.join(base_path, lang_code + "/parser.conf"))
            parser_object = BDRNNParser(config, lemmatizer_encodings, self.embeddings, runtime=True)
            parser_object.load(os.path.join(base_path, lang_code + "/parser.bestUAS"))
            self.models[PipelineComponents.PARSER] = parser_object

    def process_text(self, text="", pipeline=None):
        """
        Runs the pipeline on the input text. If the pipeline is set to None, Cube will run all available processing models
        @param text: the text to be processed. It can either be raw text format or, a list of sentences, each composed of a list of CONLLEntry
        @param pipeline: a list of PipelineComponents to be used for processing
        @return: A list of sentences, each composed of a list of CONLLEntry items
        """
        if pipeline is None:
            pipeline = [PipelineComponents.TOKENIZER, PipelineComponents.PARSER, PipelineComponents.TAGGER,
                        PipelineComponents.LEMMATIZER, PipelineComponents.COMPOUND]

        if PipelineComponents.TOKENIZER in pipeline and self.tokenizer_enabled:
            sys.stdout.write("\nTokenizing... \n\t")
            sys.stdout.flush()

            lines = text.replace("\r", "").split("\n")
            # analyze use of spaces in first part of the file
            test = "";
            useSpaces = " "
            cnt = 0
            while True:
                test = test + lines[cnt]
                # print(lines[cnt])
                if cnt + 1 >= len(lines) or cnt > 5:
                    break
                cnt += 1

            if float(test.count(' ')) / float(len(test)) < 0.02:
                useSpaces = ""
            # print (str(float(test.count(' '))/float(len(test))))
            input_string = ""
            for i in range(len(lines)):
                input_string = input_string + lines[i].replace("\r", "").replace("\n", "").strip() + useSpaces

            sequences = self.models[PipelineComponents.TOKENIZER].tokenize(input_string)

            sys.stdout.write("\n")
        else:
            sequences = text

        if PipelineComponents.COMPOUND in pipeline and self.compound_enabled:
            sequences = self.models[PipelineComponents.COMPOUND].expand_sequences(sequences)

        if PipelineComponents.PARSER in pipeline and self.parser_enabled:
            sequences = self.models[PipelineComponents.PARSER].parse_sequences(sequences)

        if PipelineComponents.TAGGER in pipeline and self.tagger_enabled:
            new_sequences = []
            for sequence in sequences:
                new_sequence = copy.deepcopy(sequence)
                predicted_tags_UPOS = self.models[PipelineComponents.TAGGER][0].tag(new_sequence)
                predicted_tags_XPOS = self.models[PipelineComponents.TAGGER][1].tag(new_sequence)
                predicted_tags_ATTRS = self.models[PipelineComponents.TAGGER][2].tag(new_sequence)
                for entryIndex in range(len(sequence)):
                    new_sequence[entryIndex].upos = predicted_tags_UPOS[entryIndex][0]
                    new_sequence[entryIndex].xpos = predicted_tags_XPOS[entryIndex][1]
                    new_sequence[entryIndex].attrs = predicted_tags_ATTRS[entryIndex][2]
                new_sequences.append(new_sequence)

            sequences = new_sequences

        if PipelineComponents.LEMMATIZER in pipeline and self.lemmatizer_enabled:
            sequences = self.models[PipelineComponents.LEMMATIZER].lemmatize_sequences(sequences)

        return sequences


if __name__ == "__main__":
    cube = Cube()
    cube.load('ro')
    sequences = cube.process_text(text="ana are mere dar nu are pere și mănâncă miere.")
    sys.stdout.write("\n\n\n")
    from io_utils.conll import Dataset

    ds = Dataset()
    ds.sequences = sequences
    ds.write_stdout()
