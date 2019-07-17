import torch.nn as nn
from cube2.networks.modules import Encoder
from cube2.networks.self_attention import SelfAttentionNetwork
from cube2.config import TaggerConfig
from cube.io_utils.encodings import Encodings


class Tagger:
    encodings: Encodings
    config: TaggerConfig

    def __init__(self, config, encodings, num_languages=1):
        super().__init__(self)
        self.config = config
        self.encodings = encodings
        self.num_languages = num_languages

        self.encoder = Encoder('float', self.config.tagger_embeddings_size, 0, self.config.tagger_encoder_size,
                               self.config.tagger_encoder_size, self.config.tagger_encoder_dropout, nn_type=nn.LSTM,
                               num_layers=self.config.tagger_encoder_layers)
        self.character_network = SelfAttentionNetwork('int', len(self.encodings.char2int),
                                                      self.config.char_input_embeddings_size,
                                                      self.config.char_encoder_size, self.config.char_encoder_layers,
                                                      self.config.tagger_embeddings_size,
                                                      self.config.tagger_encoder_dropout)
