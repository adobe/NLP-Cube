#
# Authors: Tiberiu Boros, Stefan Daniel Dumitrescu
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
import ast
from builtins import object, super
from cube.io_utils.misc import fopen
import collections

if sys.version_info[0] == 2:
    import ConfigParser
else:
    import configparser


class Config(object):
    """Generic base class that implements load/save utilities."""

    def __init__(self):
        """Call to set config object name."""
        self.__config__ = self.__class__.__name__

    def _auto_cast(self, s):
        """Autocasts string s to its original type."""
        try:
            return ast.literal_eval(s)
        except:
            return s

    def save(self, filename):
        """Save configuration to file."""
        sorted_dict = collections.OrderedDict(sorted(self.__dict__.items()))  # sort dictionary
        if sys.version_info[0] == 2:
            config = ConfigParser.ConfigParser()
        else:
            config = configparser.ConfigParser()
        config.add_section(self.__config__)  # write header
        if sys.version_info[0] == 2:
            items = sorted_dict.iteritems()
        else:
            items = sorted_dict.items()
        for k, v in items:  # for python3 use .items()
            if not k.startswith("_"):  # write only non-private properties
                if isinstance(v, float):  # if we are dealing with a float
                    str_v = str(v)
                    if "e" not in str_v and "." not in str_v:  # stop possible confusion with an int by appending a ".0"
                        v = str_v + ".0"
                v = str(v)
                config.set(self.__config__, k, v)
        with fopen(filename, 'w') as cfgfile:
            config.write(cfgfile)

    def load(self, filename):
        """Load configuration from file."""
        if sys.version_info[0] == 2:
            config = ConfigParser.ConfigParser()
        else:
            config = configparser.ConfigParser()
        config.read(filename)
        # check to see if the config file has the appropriate section
        if not config.has_section(self.__config__):
            sys.stderr.write(
                "ERROR: File \"" + filename + "\" is not a valid configuration file for the selected task: Missing section [" + self.__config__ + "]!\n")
            sys.exit(1)
        for k, v in config.items(self.__config__):
            self.__dict__[k] = self._auto_cast(v)


class TokenizerConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.cnn_filter = 512
        self.lang_emb_size = 100
        self.cnn_layers = 5
        self.external_proj_size = 300
        self.no_space_lang = False

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)

        self._valid = True


class TaggerConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.char_emb_size = 256
        self.char_filter_size = 512
        self.char_layers = 3
        self.word_emb_size = 256
        self.lang_emb_size = 64
        self.cnn_filter = 512
        self.cnn_layers = 5
        self.external_proj_size = 300
        self.lm_model = 'xlm-roberta-base'
        self._valid = True

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)


class ParserConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.char_emb_size = 256
        self.char_filter_size = 512
        self.char_layers = 5
        self.word_emb_size = 256
        self.lang_emb_size = 64
        self.cnn_filter = 512
        self.cnn_layers = 5
        self.aux_softmax_location = 5
        self.pre_parser_size = 500
        self.head_size = 100
        self.label_size = 200
        self.lm_model = 'xlm-roberta-base'
        self.external_proj_size = 300
        self.rhl_win_size = 2
        self.rnn_size = 50
        self.rnn_layers = 3

        self._valid = True

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)


class LemmatizerConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.encoder_layers = 2
        self.encoder_size = 200
        self.decoder_layers = 2
        self.decoder_size = 400
        self.att_proj_size = 100
        self.upos_emb_size = 100
        self.lang_emb_size = 100
        self.char_emb_size = 100
        self._valid = True

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)


class CompoundConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.encoder_layers = 2
        self.encoder_size = 200
        self.decoder_layers = 2
        self.decoder_size = 400
        self.att_proj_size = 100
        self.lang_emb_size = 100
        self.char_emb_size = 100
        self._valid = True

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)


class NMTConfig(Config):
    def __init__(self, filename=None):
        super().__init__()
        self.encoder_layers = [300, 300]
        self.encoder_layer_dropouts = [0.33, 0.33]
        self.decoder_layers = 2
        self.decoder_size = 300
        self.decoder_dropout = 0.33
        self.input_size = 100
        self.aux_we_layer_size = 100
        self.input_dropout_prob = 0.33

        if filename is None:
            sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)


class TieredTokenizerConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        # sentece splitting
        self.ss_char_embeddings_size = 100
        self.ss_char_peek_count = 5
        self.ss_mlp_layers = [100]
        self.ss_mlp_dropouts = [0.33]
        self.ss_lstm_size = 64
        self.ss_lstm_layers = 1
        self.ss_lstm_dropout = 0.33
        self.ss_peek_lstm_size = 64
        self.ss_peek_lstm_layers = 1
        self.ss_peek_lstm_dropout = 0.33
        # tokenization
        self.tok_char_embeddings_size = 100
        self.tok_word_embeddings_size = 100
        self.tok_mlp_layers = [100]
        self.tok_mlp_dropouts = [0.33]
        self.tok_char_lstm_layers = 2
        self.tok_char_lstm_size = 200
        self.tok_char_lstm_dropout = 0.33
        self.tok_word_lstm_layers = 2
        self.tok_word_lstm_size = 200
        self.tok_word_lstm_dropout = 0.33
        self.tok_char_peek_lstm_layers = 2
        self.tok_char_peek_lstm_size = 200
        self.tok_char_peek_lstm_dropout = 0.33

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)

        self._valid = True


class CompoundWordConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.character_embeddings_size = 100
        self.encoder_size = 200
        self.encoder_layers = 2
        self.decoder_size = 200
        self.decoder_layers = 2

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)


class GDBConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.use_char_embeddings = True
        self.char_rnn_layers = 2
        self.char_rnn_size = 100
        self.embeddings_size = 100
        self.arc_rnn_layers = [200, 200]
        self.label_rnn_size = 100
        self.proj_size = 100

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)

        self._valid = True
