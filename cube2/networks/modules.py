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
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import PackedSequence
from typing import *


class Encoder(nn.Module):
    def __init__(self, input_type, input_size, input_emb_dim, enc_hid_dim, dropout, nn_type=nn.GRU,
                 num_layers=2, ext_conditioning=0):
        super().__init__()
        assert (input_type == 'int' or input_type == 'float')
        if input_type == 'float':
            assert (input_size == input_emb_dim)
        self.input_type = input_type
        self.input_dim = input_size
        self.emb_dim = input_emb_dim
        self.enc_hid_dim = enc_hid_dim

        if self.input_type == 'int':
            self.embedding = nn.Sequential(nn.Embedding(input_size, input_emb_dim), nn.Dropout(dropout))
        else:
            self.embedding = nn.Dropout(dropout)
        if nn_type == VariationalLSTM:
            self.rnn = nn_type(input_emb_dim + ext_conditioning, enc_hid_dim, bidirectional=True, num_layers=1,
                               dropoutw=dropout, dropouto=dropout, dropouti=dropout)
            self.dropout = nn.Dropout(0)
        else:
            self.rnn = nn_type(input_emb_dim + ext_conditioning, enc_hid_dim, bidirectional=True, num_layers=1)
            self.dropout = nn.Dropout(dropout)

        if num_layers > 1:
            top_layers = []
            for ii in range(num_layers - 1):
                top_layers.append(
                    nn_type(enc_hid_dim * 2 + ext_conditioning, enc_hid_dim, bidirectional=True, num_layers=1))
            self.top_layers = nn.ModuleList(top_layers)
        else:
            self.top_layers = None

    def forward(self, src, conditioning=None):
        # src = [src sent len, batch size]
        embedded = self.embedding(src)
        # from ipdb import set_trace
        # set_trace()
        if conditioning is not None:
            conditioning = conditioning.permute(0, 1)
            conditioning = conditioning.unsqueeze(0)
            conditioning = conditioning.repeat(src.shape[0], 1, 1)
            embedded = torch.cat((embedded, conditioning), dim=2)
        # embedded = [src sent len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)
        if self.top_layers is not None:
            for rnn_layer in self.top_layers:
                if conditioning is not None:
                    outputs, hidden = rnn_layer(torch.cat((self.dropout(outputs), conditioning), dim=2))
                else:
                    outputs, hidden = rnn_layer(self.dropout(outputs))

        # outputs = [src sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        if isinstance(hidden, list) or isinstance(hidden, tuple):  # we have a LSTM
            hidden = hidden[1]
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # outputs = [src sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, return_logsoftmax=False):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src sent len, dec hid dim]
        energy = energy.permute(0, 2, 1)
        # energy = [batch size, dec hid dim, src sent len]
        # v = [dec hid dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch size, 1, dec hid dim]
        attention = torch.bmm(v, energy).squeeze(1)
        # attention= [batch size, src len]
        if return_logsoftmax:
            return F.log_softmax(attention, dim=1)
        else:
            return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, nn_type=nn.GRU, num_layers=2):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn_type((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=num_layers)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]
        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output = self.out(torch.cat((output, weighted, embedded), dim=1))
        # output = [bsz, output dim]
        return output, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        output = trg[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


# The code is adapted from https://github.com/keitakurita/Better_LSTM_PyTorch - no pip package is provided so we are
# cloning the code for testing

class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """

    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            if self.batch_first:
                max_batch_size = x.size(0)
            else:
                max_batch_size = x.size(1)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x


class VariationalLSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float = 0.,
                 dropoutw: float = 0., dropouto: float = 0.,
                 batch_first=False, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state
