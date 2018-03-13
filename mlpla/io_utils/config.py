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

import sys
import json
import os.path
from builtins import object, super
        
class Config(object):
    def __init__ (self):
        self.__config__ = self.__class__.__name__
        
    def save (self, filename):            
        # TODO hide fields like "valid", etc
        json.dump(self.__dict__, open(filename, 'w'), sort_keys=True, indent=4)          
        
    def load (self, filename):
        if not os.path.isfile(filename):
            raise IOError("Configuration file \""+filename+"\" not found.")         
        # TODO check that __config__ == self.__class__.__name__ so we don't load another config
        self.__dict__ = json.load(open(filename, 'r'))
        #print(self.__dict__)    

class TokenizerConfig (Config):
    def __init__(self, filename=None):
        super().__init__()
        sys.stdout.write("Reading configuration file... ")

        self.base = ""
        # encoder-char        
        self.char_vocabulary_size = -1  # to be calculated when first training
        self.char_embedding_size = 100
        self.char_generic_feature_vocabulary_size = 2
        self.char_generic_feature_embedding_size = 5

        self.encoder_char_input_attribute_dropout = 0.
        self.encoder_char_input_noise = 0.
        self.encoder_char_blstm_size = 100
        self.encoder_char_lstm_size = 200

        # next-chars
        self.next_chars_embedding_size = 100
        self.next_chars_window_size = 10

        # encoder-word        
        self.encoder_word_input_w2i_array = {}
        self.encoder_word_vocab_size = 0  # ref
        self.encoder_word_embedding_size = 0  # ref
        self.encoder_word_lstm_size = 200
        # decoder 
        self.decoder_attribute_dropout = 0.33
        self.decoder_hidden_size = 20

        self.dropout_rate = 0
        # extra        
        self.patience = -1
        self.tokenize_maximum_sequence_length = 500  # how much to run predict on, at a time

        if filename == None:
            sys.stdout.write("no configuration file supplied. Using default values\n")
        else: 
            sys.stdout.write("reading configuration file ["+filename+"]\n")
            self.load(filename)
            
        self.valid = True


class TaggerConfig (Config):
    def __init__(self, filename=None):
        super().__init__()
        sys.stdout.write("Reading configuration file... ")
        self.layers = [200, 200]
        self.layer_dropouts = [0.5, 0.5]
        self.aux_softmax_layer = 1
        self.valid = True
        self.input_dropout_prob = 0.33
        self.presoftmax_mlp_layers = [500]
        self.presoftmax_mlp_dropouts = [0.5]
        self.input_size = 100

        if filename == None:
            sys.stdout.write("no configuration file supplied. Using default values\n")
        else: 
            sys.stdout.write("reading configuration file ["+filename+"]\n")
            self.load(filename)
            
        print "INPUT SIZE:", self.input_size
        print "LAYERS:", self.layers
        print "LAYER DROPOUTS:", self.layer_dropouts
        print "AUX SOFTMAX POSITION:", self.aux_softmax_layer
        print "INPUT DROPOUT PROB:", self.input_dropout_prob
        print "PRESOFTMAX MLP LAYERS:", self.presoftmax_mlp_layers
        print "PRESOFTMAX MLP DROPOUT:", self.presoftmax_mlp_dropouts

        if self.aux_softmax_layer > len(self.layers) - 1 or self.aux_softmax_layer == 0:
            print "Configuration error: aux softmax layer must be placed after the first layer and before the final one"
            self.valid = False


class ParserConfig (Config):
    def __init__(self, filename=None):
        super().__init__()    
        sys.stdout.write("Reading configuration file... ")
        self.layers = [300, 300, 50, 50, 50]
        self.layer_dropouts = [0.33, 0.33, 0.33, 0.33, 0.33]
        self.aux_softmax_layer = 2
        self.valid = True
        self.input_dropout_prob = 0.33
        self.arc_proj_size = 100
        self.label_proj_size = 400
        self.presoftmax_mlp_dropout = 0.33
        self.predict_morphology = True
        self.use_morphology = False
        self.input_embeddings_size = 100

        if filename == None:
            sys.stdout.write("no configuration file supplied. Using default values\n")
        else: 
            sys.stdout.write("reading configuration file ["+filename+"]\n")
            self.load(filename)
            
        print "LAYERS:", self.layers
        print "LAYER DROPOUTS:", self.layer_dropouts
        print "AUX SOFTMAX POSITION:", self.aux_softmax_layer
        print "INPUT DROPOUT PROB:", self.input_dropout_prob
        print "ARC PROJECTION SIZE:", self.arc_proj_size
        print "LABEL PROJECTION SIZE:", self.label_proj_size
        print "PRESOFTMAX MLP DROPOUT:", self.presoftmax_mlp_dropout
        print "JOINTLY PARSE AND PREDICT MORPHOLOGY:", self.predict_morphology
        print "USE MORPHOLOGY AS INPUT:", self.use_morphology
        print "INPUT EMBEDDINGS SIZE:", self.input_embeddings_size

        if self.aux_softmax_layer > len(self.layers) - 1 or self.aux_softmax_layer == 0:
            print "Configuration error: aux softmax layer must be placed after the first layer and before the final one"
            self.valid = False

        if self.use_morphology and self.predict_morphology:
            print "Configuration error: you are using morphology to predict morphology."
            self.valid = False

    
class LemmatizerConfig (Config):
    def __init__(self, filename=None):
        super().__init__()
        self.rnn_size = 500
        self.rnn_layers = 2
        self.char_embeddings = 100
        self.char_rnn_size = 200
        self.char_rnn_layers = 1
        self.tag_embeddings_size = 100
        
        if filename == None:
            sys.stdout.write("no configuration file supplied. Using default values\n")
        else: 
            sys.stdout.write("reading configuration file ["+filename+"]\n")
            self.load(filename)


class NMTConfig (Config):
    def __init__(self, filename=None):
        super().__init__()
        self.encoder_layers = [300, 300]
        self.encoder_layer_dropouts = [0.33, 0.33]
        self.decoder_layers = 2
        self.decoder_size = 300
        self.decoder_dropout=0.33
        self.input_size = 100
        self.aux_we_layer_size = 100
        self.input_dropout_prob = 0.33
        
        if filename == None:
            sys.stdout.write("no configuration file supplied. Using default values\n")
        else: 
            sys.stdout.write("reading configuration file ["+filename+"]\n")
            self.load(filename)
