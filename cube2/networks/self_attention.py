import torch
import torch.nn as nn
from cube2.networks.modules import Encoder
from cube2.networks.modules import Attention


class SelfAttentionNetwork(nn.Module):
    def __init__(self, input_type, input_size, input_emb_size, encoder_size, encoder_layers, output_size, dropout):
        super(SelfAttentionNetwork, self).__init__()
        self.encoder = Encoder(input_type, input_size, input_emb_size, encoder_size, output_size, dropout,
                               nn_type=nn.GRU, num_layers=encoder_layers)

        self.attention = Attention(encoder_size, encoder_size)

    def forward(self, x):
        output, hidden = self.encoder(x)
        attention = self.attention(output.squeeze(0)[-1], hidden)
        return attention
