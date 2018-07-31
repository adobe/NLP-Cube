# -*- coding: utf-8 -*-

import sys
import os
from io_utils.encodings import Encodings
from io_utils.embeddings import WordEmbeddings
from io_utils.model_store import ModelStore, PipelineComponents


class Cube(object):
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
        self.model_store = ModelStore()

    def load(self, lang_code, check_for_latest=True):
        """
        Loads the pipeline with all available models for the target language.

        @param lang_code: Target language code. See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
        @param check_for_latest: True if we always want the latest model.
        """
        self.model_store.load(lang_code, check_for_latest)

        # Load models from the ModelStore.
        self.models = self.model_store.models

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
                import copy
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
    sequences = cube.process_text(
        text="Ana are mere, și Maria are pere. Ce mai faci băiatule, deși mi se pare că ai peri deși. "
             "Și-a făcut casă noaptea și-a plecat acasă. Ce faci bosule? Ce faci boule? De ce terminația asta nu e de "
             "substantiv masculin?"
             "Ți-am zădărnicit șerpicoarea maricoasă că să veremești frumos de tot. "
             "Stadioanele sunt ocupate de copii fotbaliști și de copii talentați la sport.")
    sys.stdout.write("\n\n\n")
    from io_utils.conll import Dataset

    ds = Dataset()
    ds.sequences = sequences
    ds.write_stdout()
