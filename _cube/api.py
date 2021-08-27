# -*- coding: utf-8 -*-

import sys
import os

class Cube(object):
    def __init__(self, verbose=False, random_seed=None, memory=512, autobatch=False, use_gpu=False):
        """
        Create an empty instance for Cube
        Before it can be used, you must call @method load with @param language_code set to your target language        
        """
        self._loaded = False
        self._verbose = verbose           
        import dynet_config
        
        if random_seed != None:
            if not isinstance(random_seed, int):
                raise Exception ("Random seed must be an integer!")   
            if random_seed == 0:
                print("[Warning] While Python and Numpy's seeds are now set to 0, DyNet uses 0 to reset the seed generator (fully random). Use any non-zero int value to set DyNet to a fixed random seed.")
            # set python random seed
            import random
            random.seed(random_seed)
            #set numpy random seed
            import numpy as np
            np.random.seed(random_seed)
        else:
            random_seed = 0 # this is the default value for DyNet (meaning full random)        
            
        dynet_config.set(mem=memory, random_seed=random_seed, autobatch=autobatch)
        if use_gpu:
            dynet_config.set_gpu()        

    def load(self, language_code, version="latest", local_models_repository=None, local_embeddings_file=None, tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True):
        """
        Loads the pipeline with all available models for the target language.

        @param lang_code: Target language code. See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
        @param version: "latest" to get the latest version, or other specific version in like "1.0", "2.1", etc .
       
        """
        from .io_utils.encodings import Encodings
        from .io_utils.embeddings import WordEmbeddings
        from .io_utils.model_store import ModelMetadata, ModelStore
        from .io_utils.config import TieredTokenizerConfig, CompoundWordConfig, LemmatizerConfig, TaggerConfig, ParserConfig
        from .generic_networks.tokenizers import TieredTokenizer
        from .generic_networks.token_expanders import CompoundWordExpander
        from .generic_networks.lemmatizers import FSTLemmatizer
        from .generic_networks.taggers import BDRNNTagger
        from .generic_networks.parsers import BDRNNParser        
        
        self._tokenizer = None  # tokenizer object, default is None
        self._compound_word_expander = None  # compound word expander, default is None
        self._lemmatizer = None  # lemmatizer object, default is None
        self._parser = None  # parser object, default is None
        self._tagger = None  # tagger object, default is None
        self.metadata = ModelMetadata()
        
        # Initialize a ModelStore object
        if local_models_repository: 
            model_store_object = ModelStore(disk_path=local_models_repository)
        else: 
            model_store_object = ModelStore()

        # Find a local model or download it if it does not exist, returning the local model folder path
        model_folder_path = model_store_object.find(lang_code=language_code, version=version, verbose=self._verbose)

        # If the model contains metadata, load it        
        if os.path.isfile(os.path.join(model_folder_path, "metadata.json")): 
            self.metadata.read(os.path.join(model_folder_path, "metadata.json"))
        else:
            self.metadata = None

        # Load embeddings                
        embeddings = WordEmbeddings(verbose=False)
        if self._verbose:
            sys.stdout.write('\tLoading embeddings ... \n')
        if local_embeddings_file is not None:        
            embeddings.read_from_file(local_embeddings_file, None, full_load=False)
        else: # embeddings file is not manually specified
            if self.metadata is None: # no metadata exists
                raise Exception("When using a locally-trained model please specify a path to a local embeddings file (local_embeddings_file cannot be None).")    
            else: # load from the metadata path
                if self.metadata.embeddings_file_name is None or self.metadata.embeddings_file_name == "":
                    # load a dummy embedding
                    embeddings.load_dummy_embeddings()
                else:
                    # load full embedding from file
                    emb_path = os.path.join(model_store_object.embeddings_repository, self.metadata.embeddings_file_name)
                    if not os.path.exists(emb_path):
                        raise Exception("Embeddings file not found: {}".format(emb_path))
                    embeddings.read_from_file(emb_path, None, full_load=False)
       
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

        # 2. Load compound
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

        # 3. Load lemmatizer
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

        # 4. Load tagger
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

        # 5. Load parser
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
            if not isinstance(text, str):
                raise Exception("The text argument must be a string!")
            # split text by lines
            input_lines = text.split("\n")
            for input_line in input_lines:                
                sequences+=self._tokenizer.tokenize(input_line)                
        else:
            if not isinstance(text, list):
                raise Exception("The text argument must be a list of lists of tokens!")
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
    cube = Cube(verbose=True)
    cube.load('en', tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True)
    cube.metadata.info()

    text = "I'm a success today because I had a friend who believed in me and I didn't have the heart to let him down. This is a quote by Abraham Lincoln."

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
