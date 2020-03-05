import sys
import ast
from builtins import object, super
from cube.misc.misc import fopen
import collections
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


class TaggerConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.tagger_embeddings_size = 100
        self.tagger_encoder_size = 500
        self.tagger_encoder_layers = 2
        self.tagger_encoder_dropout = 0.33
        self.tagger_input_dropout_prob = 0.3
        self.tagger_mlp_layer = 500
        self.tagger_mlp_dropout = 0.3
        self.char_encoder_size = 200
        self.char_encoder_layers = 2
        self.char_input_embeddings_size = 100
        self.aux_softmax_layer_index = 1
        self.aux_softmax_weight = 0
        self._valid = True

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)

        if verbose:
            print("INPUT SIZE:", self.input_size)
            print("LAYERS:", self.layers)
            print("LAYER DROPOUTS:", self.layer_dropouts)
            print("AUX SOFTMAX POSITION:", self.aux_softmax_layer)
            print("INPUT DROPOUT PROB:", self.input_dropout_prob)
            print("PRESOFTMAX MLP LAYERS:", self.presoftmax_mlp_layers)
            print("PRESOFTMAX MLP DROPOUT:", self.presoftmax_mlp_dropouts)


class ParserConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.tagger_embeddings_size = 100
        self.tagger_encoder_size = 500
        self.tagger_encoder_layers = 5
        self.tagger_encoder_dropout = 0.33
        self.tagger_input_dropout_prob = 0.33
        self.tagger_mlp_layer = 500
        self.tagger_mlp_dropout = 0.33
        self.parser_arc_proj_size = 100
        self.parser_label_proj_size = 400
        self.char_encoder_size = 200
        self.char_encoder_layers = 2
        self.char_input_embeddings_size = 100
        self.aux_softmax_layer_index = 2
        self.aux_softmax_weight = 0.01
        self._valid = True

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)

        if verbose:
            print("INPUT SIZE:", self.input_size)
            print("LAYERS:", self.layers)
            print("LAYER DROPOUTS:", self.layer_dropouts)
            print("AUX SOFTMAX POSITION:", self.aux_softmax_layer)
            print("INPUT DROPOUT PROB:", self.input_dropout_prob)
            print("PRESOFTMAX MLP LAYERS:", self.presoftmax_mlp_layers)
            print("PRESOFTMAX MLP DROPOUT:", self.presoftmax_mlp_dropouts)


class TokenizerConfig(Config):
    def __init__(self, filename=None, verbose=False):
        super().__init__()
        self.char_emb_size = 100
        self.conv_layers = 5
        self.conv_kernel = 5
        self.conv_filters = 512
        self.lang_emb_size = 100
        self.rnn_size = 300
        self.rnn_layers = 2
        self._valid = True

        if filename is None:
            if verbose:
                sys.stdout.write("No configuration file supplied. Using default values.\n")
        else:
            if verbose:
                sys.stdout.write("Reading configuration file " + filename + " \n")
            self.load(filename)
