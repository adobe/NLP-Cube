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
import pytorch_lightning as pl


class LinearNorm(pl.LightningModule):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_normal_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=500, dropout=0.33, hid_func=torch.tanh):
        super().__init__()
        self._h1 = LinearNorm(in_dim, hid_dim)
        self._h2 = LinearNorm(hid_dim, out_dim)
        self._dropout = dropout
        self._hid_func = hid_func

    def forward(self, x):
        h = torch.dropout(self._hid_func(self._h1(x)), self._dropout, self.training)
        o = self._h2(h)
        return o


class Encoder(pl.LightningModule):
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
                               dropoutw=dropout, dropouto=dropout, dropouti=dropout, batch_first=True)
            self.dropout = nn.Identity()
        else:
            self.rnn = nn_type(input_emb_dim + ext_conditioning, enc_hid_dim, bidirectional=True, num_layers=1,
                               batch_first=True)
            self.dropout = nn.Dropout(dropout)

        if num_layers > 1:
            top_layers = []
            for ii in range(num_layers - 1):
                if nn_type == VariationalLSTM:
                    top_layers.append(
                        nn_type(enc_hid_dim * 2 + ext_conditioning, enc_hid_dim, bidirectional=True, num_layers=1,
                                batch_first=True, dropoutw=dropout, dropouto=dropout, dropouti=dropout))
                else:
                    top_layers.append(
                        nn_type(enc_hid_dim * 2 + ext_conditioning, enc_hid_dim, bidirectional=True, num_layers=1,
                                batch_first=True))

            self.top_layers = nn.ModuleList(top_layers)
        else:
            self.top_layers = None

    def forward(self, src, conditioning=None):
        embedded = self.embedding(src)

        if conditioning is not None:
            # conditioning = conditioning.permute(0, 1)
            conditioning = conditioning.unsqueeze(1)
            conditioning = conditioning.repeat(1, src.shape[1], 1)
            embedded = torch.cat((embedded, conditioning), dim=2)

        outputs, hidden = self.rnn(embedded)
        if self.top_layers is not None:
            for rnn_layer in self.top_layers:
                if conditioning is not None:
                    outputs, hidden = rnn_layer(torch.cat((self.dropout(outputs), conditioning), dim=2))
                else:
                    outputs, hidden = rnn_layer(self.dropout(outputs))
        if isinstance(hidden, list) or isinstance(hidden, tuple):  # we have a LSTM
            hidden = hidden[1]
        hidden = torch.cat((hidden[-1, :, :], hidden[0, :, :]), dim=1)

        return outputs, hidden


class BilinearAttention(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.biliniar = nn.Bilinear(dim1, dim2, 1)
        self.linear = nn.Linear(dim1 + dim2, 1)

    def forward(self, query, keys):
        query = query.unsqueeze(1).repeat(1, keys.shape[1], 1)
        biliniar = self.biliniar(query, keys)
        h = torch.cat([query, keys], dim=-1)
        liniar = self.linear(h)
        return (liniar + biliniar).squeeze(2)


class Attention(pl.LightningModule):
    def __init__(self, enc_hid_dim, dec_hid_dim, att_proj_size=100):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        # self.attn = LinearNorm((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.attn = ConvNorm(enc_hid_dim * 2 + dec_hid_dim, att_proj_size, kernel_size=5,
                             w_init_gain='tanh')
        self.v = nn.Parameter(torch.rand(att_proj_size))

    def forward(self, hidden, encoder_outputs, return_logsoftmax=False):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs
        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        energy = torch.dropout(
            torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2).transpose(1, 2)).transpose(1, 2)), 0.1,
            self.training)
        energy = energy.transpose(1, 2)
        # energy = [batch size, src sent len, dec hid dim]
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


class Decoder(pl.LightningModule):
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
        self.out = LinearNorm((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output = self.out(torch.cat((output, weighted, embedded), dim=1))
        return output, hidden.squeeze(0)


class Seq2Seq(pl.LightningModule):
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

class VariationalDropout(pl.LightningModule):
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


class VariationalLSTM(pl.LightningModule):
    def __init__(self, *args, dropouti: float = 0.,
                 dropoutw: float = 0., dropouto: float = 0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
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
        self.flatten_parameters()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


class ConvNorm(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_normal_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


# class WordGram(pl.LightningModule):
#     def __init__(self, num_chars: int, num_langs: int, num_filters=512, char_emb_size=256, case_emb_size=32,
#                  lang_emb_size=32, num_layers=3):
#         super(WordGram, self).__init__()
#         NUM_FILTERS = num_filters
#         self._num_filters = NUM_FILTERS
#         self._lang_emb = nn.Embedding(num_langs + 1, lang_emb_size)
#         self._tok_emb = nn.Embedding(num_chars + 1, char_emb_size)
#         self._case_emb = nn.Embedding(4, case_emb_size)
#         self._num_layers = num_layers
#
#         convolutions_char = []
#         cs_inp = char_emb_size + lang_emb_size + case_emb_size
#         for _ in range(num_layers):
#             conv_layer = nn.Sequential(
#                 ConvNorm(cs_inp,
#                          NUM_FILTERS,
#                          kernel_size=5, stride=1,
#                          padding=2,
#                          dilation=1, w_init_gain='tanh'),
#                 nn.BatchNorm1d(NUM_FILTERS))
#             convolutions_char.append(conv_layer)
#             cs_inp = NUM_FILTERS // 2 + lang_emb_size
#         self._convolutions_char = nn.ModuleList(convolutions_char)
#
#         self._rnn = nn.LSTM(NUM_FILTERS // 2, NUM_FILTERS // 2, num_layers=2)
#         self._pre_out = LinearNorm(NUM_FILTERS // 2, NUM_FILTERS // 2)
#
#     def forward(self, x_char, x_case, x_lang, x_mask, x_word_len):
#         x_char = self._tok_emb(x_char)
#         x_case = self._case_emb(x_case)
#         x_lang = self._lang_emb(x_lang)
#
#         x = torch.cat([x_char, x_case], dim=-1)
#         x = x.permute(0, 2, 1)
#         x_lang = x_lang.unsqueeze(1).repeat(1, x_case.shape[1], 1).permute(0, 2, 1)
#         half = self._num_filters // 2
#         count = 0
#         res = None
#         skip = None
#         for conv in self._convolutions_char:
#             count += 1
#             drop = self.training
#             if count >= len(self._convolutions_char):
#                 drop = False
#             if skip is not None:
#                 x = x + skip
#
#             x = torch.cat([x, x_lang], dim=1)
#             conv_out = conv(x)
#             tmp = torch.tanh(conv_out[:, :half, :]) * torch.sigmoid((conv_out[:, half:, :]))
#             if res is None:
#                 res = tmp
#             else:
#                 res = res + tmp
#             skip = tmp
#             x = torch.dropout(tmp, 0.1, drop)
#         x = x + res
#         x = x.permute(0, 2, 1)
#         x = torch.flip(x, dims=[1])
#         out, _ = self._rnn(x)
#         norm = out[:, -1, :]
#
#         return torch.tanh(self._pre_out(norm))
#
#     def _get_device(self):
#         if self._lang_emb.weight.device.type == 'cpu':
#             return 'cpu'
#         return '{0}:{1}'.format(self._lang_emb.weight.device.type, str(self._lang_emb.weight.device.index))
#
#     def save(self, path):
#         torch.save(self.state_dict(), path)
#
#     def load(self, path):
#         self.load_state_dict(torch.load(path, map_location='cpu'))


class WordGram(pl.LightningModule):
    def __init__(self, num_chars: int, num_langs: int, num_filters=512, char_emb_size=256, case_emb_size=32,
                 lang_emb_size=32, num_layers=3):
        super(WordGram, self).__init__()
        NUM_FILTERS = num_filters
        self._num_filters = NUM_FILTERS
        self._lang_emb = nn.Embedding(num_langs + 1, lang_emb_size)
        self._tok_emb = nn.Embedding(num_chars + 1, char_emb_size)
        self._case_emb = nn.Embedding(4, case_emb_size)
        self._num_layers = num_layers
        convolutions_char = []
        cs_inp = char_emb_size + lang_emb_size + case_emb_size
        for _ in range(num_layers):
            conv_layer = nn.Sequential(
                ConvNorm(cs_inp,
                         NUM_FILTERS,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(NUM_FILTERS))
            convolutions_char.append(conv_layer)
            cs_inp = NUM_FILTERS // 2 + lang_emb_size
        self._convolutions_char = nn.ModuleList(convolutions_char)
        self._pre_out = LinearNorm(NUM_FILTERS // 2, NUM_FILTERS // 2)

    def forward(self, x_char, x_case, x_lang, x_mask, x_word_len):
        x_char = self._tok_emb(x_char)
        x_case = self._case_emb(x_case)
        x_lang = self._lang_emb(x_lang)

        x = torch.cat([x_char, x_case], dim=-1)
        x = x.permute(0, 2, 1)
        x_lang = x_lang.unsqueeze(1).repeat(1, x_case.shape[1], 1).permute(0, 2, 1)
        half = self._num_filters // 2
        count = 0
        res = None
        skip = None
        for conv in self._convolutions_char:
            count += 1
            drop = self.training
            if count >= len(self._convolutions_char):
                drop = False
            if skip is not None:
                x = x + skip

            x = torch.cat([x, x_lang], dim=1)
            conv_out = conv(x)
            tmp = torch.tanh(conv_out[:, :half, :]) * torch.sigmoid((conv_out[:, half:, :]))
            if res is None:
                res = tmp
            else:
                res = res + tmp
            skip = tmp
            x = torch.dropout(tmp, 0.1, drop)
        x = x + res
        x = x.permute(0, 2, 1)
        x = x * x_mask.unsqueeze(2)
        pre = torch.sum(x, dim=1, dtype=torch.float)
        norm = pre / x_word_len.unsqueeze(1)
        return torch.tanh(self._pre_out(norm))

    def _get_device(self):
        if self._lang_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._lang_emb.weight.device.type, str(self._lang_emb.weight.device.index))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


'''
Code adapted from https://github.com/stanfordnlp/stanza/
'''


class PairwiseBilinear(nn.Module):
    ''' A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)'''

    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.Tensor(input1_size, input2_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size)) if bias else 0

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        intermediate = torch.mm(input1.view(-1, input1_size[-1]),
                                self.weight.view(-1, self.input2_size * self.output_size))
        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)

        return output


class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class PairwiseBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class DeepBiaffine(nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, output_size, hidden_func=F.relu, dropout=0,
                 pairwise=True):
        super().__init__()
        self.W1 = nn.Linear(input1_size, hidden_size)
        self.W2 = nn.Linear(input2_size, hidden_size)
        self.hidden_func = hidden_func
        if pairwise:
            self.scorer = PairwiseBiaffineScorer(hidden_size, hidden_size, output_size)
        else:
            self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))),
                           self.dropout(self.hidden_func(self.W2(input2))))
