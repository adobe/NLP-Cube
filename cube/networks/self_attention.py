#
# Author: Tiberiu Boros
#
# Copyright (c) 2019 Adobe Systems Incorporated. All rights reserved.
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

import torch
import torch.nn as nn
from cube.networks.modules import Attention, LinearNorm, Encoder, ConvNorm


class SelfAttentionNetwork(nn.Module):
    def __init__(self, input_type, input_size, input_emb_size, encoder_size, encoder_layers, output_size, dropout,
                 nn_type=nn.GRU, ext_conditioning=0):
        super(SelfAttentionNetwork, self).__init__()
        self.input_type = input_type
        self.encoder = Encoder(input_type, 512, 512, encoder_size, dropout,
                               nn_type=nn_type, num_layers=encoder_layers, ext_conditioning=ext_conditioning)
        self.encoder_dropout = nn.Dropout(dropout)

        self.attention = Attention(encoder_size, encoder_size * 2)
        self.mlp = LinearNorm(encoder_size * 4 + ext_conditioning, output_size)

        convolutions = []
        cs_inp = input_emb_size
        for _ in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(cs_inp,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
            cs_inp = 512
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x, conditioning=None):
        # batch_size should be the second column for whatever reason
        x = x.permute(0, 2, 1)
        for conv in self.convolutions:
            x = torch.dropout(torch.relu(conv(x)), 0.5, self.training)

        x = x.permute(0, 2, 1)

        output, hidden = self.encoder(x, conditioning=conditioning)
        output = self.encoder_dropout(output)
        hidden = self.encoder_dropout(hidden)
        attention = self.attention(hidden, output)

        encoder_outputs = output
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        weighted = torch.bmm(attention.unsqueeze(1), encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2]

        pre_mlp = torch.cat([weighted, hidden.unsqueeze(1), conditioning.unsqueeze(1)], dim=2)
        return torch.tanh(self.mlp(pre_mlp))
