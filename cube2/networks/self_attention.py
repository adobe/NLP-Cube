import torch
import torch.nn as nn
from cube2.networks.modules import Encoder
from cube2.networks.modules import Attention


class SelfAttentionNetwork(nn.Module):
    def __init__(self, input_type, input_size, input_emb_size, encoder_size, encoder_layers, output_size, dropout,
                 nn_type=nn.GRU, ext_conditioning=0):
        super(SelfAttentionNetwork, self).__init__()
        self.input_type = input_type
        self.encoder = Encoder(input_type, input_size, input_emb_size, encoder_size, output_size, dropout,
                               nn_type=nn_type, num_layers=encoder_layers, ext_conditioning=ext_conditioning)
        self.encoder_dropout = nn.Dropout(dropout)

        self.attention = Attention(output_size // 2, encoder_size * 2)
        self.mlp = nn.Linear(encoder_size * 2 + output_size, output_size)

    def forward(self, x, conditioning=None):
        # batch_size should be the second column for whatever reason
        if self.input_type == 'int':
            x = x.permute(1, 0)
        else:
            x = x.permute(1, 0, 2)
        output, hidden = self.encoder(x, conditioning=conditioning)
        output = self.encoder_dropout(output)
        hidden = self.encoder_dropout(hidden)
        attention = self.attention(hidden, output)

        encoder_outputs = output.permute(1, 0, 2)
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        weighted = torch.bmm(attention.unsqueeze(1), encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        pre_mlp = torch.cat([weighted, hidden.unsqueeze(0)], dim=2)
        return self.mlp(pre_mlp)
