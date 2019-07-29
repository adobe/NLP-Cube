import sys, os
sys.path.append("../../..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from cube2.components.lookup import Lookup

class Encoder(nn.Module):
    def __init__(self, input_type, input_size, input_emb_dim, enc_hid_dim, output_dim, dropout, nn_type=nn.GRU,
                 num_layers=2):
        super().__init__()
        assert (input_type == 'int' or input_type == 'float')
        self.input_type = input_type
        self.input_dim = input_size
        self.emb_dim = input_emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = output_dim
        self.dropout = dropout

        if self.input_type == 'int':
            self.embedding = nn.Embedding(input_size, input_emb_dim)
        else:
            self.embedding = nn.Sequential(nn.Linear(input_size, input_emb_dim), nn.Tanh())

        self.rnn = nn_type(input_emb_dim, enc_hid_dim, bidirectional=True, num_layers=num_layers, dropout=dropout)

        self.fc = nn.Linear(enc_hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src sent len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)
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
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
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

    def forward(self, hidden, encoder_outputs):
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
        return F.softmax(attention, dim=1)

class SelfAttentionNetwork(nn.Module):
    def __init__(self, input_type, input_size, input_emb_size, encoder_size, encoder_layers, output_size, dropout,
                 nn_type=nn.GRU):
        super(SelfAttentionNetwork, self).__init__()
        self.input_type = input_type
        self.encoder = Encoder(input_type, input_size, input_emb_size, encoder_size, output_size, dropout,
                               nn_type=nn_type, num_layers=encoder_layers)
        self.encoder_dropout = nn.Dropout(dropout)

        self.attention = Attention(output_size // 2, encoder_size * 2)
        self.mlp = nn.Linear(encoder_size * 2 + output_size, output_size)

    def forward(self, x):
        # batch_size should be the second column for whatever reason
        if self.input_type == 'int':
            x = x.permute(1, 0)
        else:
            x = x.permute(1, 0, 2)
        output, hidden = self.encoder(x)
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

class TextEncoder(nn.Module):
    lookup: Lookup

    def __init__(self, lookup, input_dropout_prob, char_encoder_size, char_encoder_layers, char_input_embeddings_size, input_size, total_layers, aux_softmax_layer_index, output_size, mlp_output_size, dropout, ext_conditioning=None, device='cpu'):
        super(TextEncoder, self).__init__()
        self.lookup = lookup
        
        self.input_dropout_prob = input_dropout_prob
        self.char_encoder_size = char_encoder_size
        self.char_encoder_layers = char_encoder_layers
        self.char_input_embeddings_size = char_input_embeddings_size 
        self.input_size = input_size # self.config.tagger_embeddings_size
        self.total_layers = total_layers
        self.aux_softmax_layer_index = aux_softmax_layer_index
        self.output_size = output_size
        self.mlp_output_size = mlp_output_size
        self.dropout = dropout
        
        self.use_conditioning = (ext_conditioning is None)
        if ext_conditioning is None:
            ext_conditioning = 0
        self._device = device

        self.first_encoder = Encoder('float', self.input_size * 2 + ext_conditioning,
                                     self.input_size,
                                     self.output_size,
                                     self.output_size, self.dropout,
                                     nn_type=nn.LSTM,
                                     num_layers=self.aux_softmax_layer_index)
        self.second_encoder = Encoder('float', self.output_size * 2,
                                      self.input_size,
                                      self.output_size,
                                      self.output_size, self.dropout,
                                      nn_type=nn.LSTM,
                                      num_layers=self.total_layers - self.aux_softmax_layer_index)
        self.character_network = SelfAttentionNetwork('float', self.char_input_embeddings_size,
                                                      self.char_input_embeddings_size,
                                                      self.char_encoder_size, self.char_encoder_layers,
                                                      self.input_size,
                                                      self.dropout, nn_type=nn.LSTM)

        mlp_input_size = self.output_size * 2 + ext_conditioning
        self.mlp = nn.Sequential(nn.Linear(mlp_input_size, self.mlp_output_size, bias=True),
                                 nn.Tanh(),
                                 nn.Dropout(p=self.dropout))

        self.word_emb = nn.Embedding(len(self.lookup.word2int), self.input_size, padding_idx=0)
        self.char_emb = nn.Embedding(len(self.lookup.char2int), self.char_input_embeddings_size,
                                     padding_idx=0)
        self.case_emb = nn.Embedding(5, 16,
                                     padding_idx=0)
        self.encoder_dropout = nn.Dropout(p=self.dropout)

        self.char_proj = nn.Linear(self.char_input_embeddings_size + 16, self.char_input_embeddings_size)

        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and "rnn" in name:  # forget bias
                nn.init.zeros_(param.data)
                param.data[param.size()[0] // 4:param.size()[0] // 2] = 1

    def forward(self, x, conditioning=None):
    
        (lang_id_sequences_tensor, seq_lengths, word_sequences_tensor, char_sequences_tensor, symbol_sequences_tensor, seq_masks, char_seq_lengths, symbol_seq_lengths, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor) = x
        
    
        char_network_batch, word_network_batch = self._create_batches(x)
        char_network_output = self.character_network(char_network_batch)
        word_emb = self.word_emb(word_network_batch)
        char_emb = char_network_output.view(word_emb.size())
        if self.training:
            masks_char, masks_word = self._compute_masks(char_emb.size(), self.input_dropout_prob)
            x = torch.cat(
                (torch.tanh(masks_char.unsqueeze(2)) * char_emb, torch.tanh(masks_word.unsqueeze(2)) * word_emb), dim=2)
        else:
            x = torch.cat((torch.tanh(char_emb), torch.tanh(word_emb)), dim=2)
        output_hidden, hidden = self.first_encoder(x.permute(1, 0, 2))
        output_hidden = self.encoder_dropout(output_hidden)
        output, hidden = self.second_encoder(output_hidden)
        output = self.encoder_dropout(output)
        return self.mlp(output.permute(1, 0, 2)), output_hidden.permute(1, 0, 2)

    def _compute_masks(self, size, prob):
        m1 = np.ones(size[:-1])
        m2 = np.ones(size[:-1])

        for ii in range(m1.shape[0]):
            for jj in range(m2.shape[1]):
                p1 = random.random()
                p2 = random.random()
                if p1 >= prob and p2 < prob:
                    mm1 = 2
                    mm2 = 0
                elif p1 < prob and p2 >= prob:
                    mm1 = 0
                    mm2 = 2
                elif p1 < prob and p2 < prob:
                    mm1 = 0
                    mm2 = 0
                else:
                    mm1 = 1
                    mm2 = 1
                m1[ii, jj] = mm1
                m2[ii, jj] = mm2
        device = self._get_device()
        return torch.tensor(m1, dtype=torch.float32, device=device), torch.tensor(m2, dtype=torch.float32,
                                                                                  device=device)

    @staticmethod
    def _case_index(char):
        if char.lower() == char.upper():  # symbol
            return 3
        elif char.upper() != char:  # lowercase
            return 2
        else:  # uppercase
            return 1

    def _get_device(self):
        return self._device

    def _create_batches(self, x):
        char_batch = []
        case_batch = []
        word_batch = []
        max_sent_size = 0
        max_word_size = 0

        for sent in x:
            if len(sent) > max_sent_size:
                max_sent_size = len(sent)
            for entry in sent:
                if len(entry.word) > max_word_size:
                    max_word_size = len(entry.word)
        # print(max_sent_size)
        for sent in x:
            sent_int = []

            for entry in sent:
                char_int = []
                case_int = []
                if entry.word.lower() in self.lookup.word2int:
                    sent_int.append(self.lookup.word2int[entry.word.lower()])
                else:
                    sent_int.append(self.lookup.word2int['<UNK>'])
                for char in entry.word:
                    if char.lower() in self.lookup.char2int:
                        char_int.append(self.lookup.char2int[char.lower()])
                    else:
                        char_int.append(self.lookup.char2int['<UNK>'])
                    case_int.append(self._case_index(char))
                for _ in range(max_word_size - len(entry.word)):
                    char_int.append(self.lookup.char2int['<PAD>'])
                    case_int.append(0)

                char_batch.append(char_int)
                case_batch.append(case_int)

            for _ in range(max_sent_size - len(sent)):
                sent_int.append(self.lookup.word2int['<PAD>'])
                char_batch.append([0 for _ in range(max_word_size)])
                case_batch.append([0 for _ in range(max_word_size)])
            word_batch.append(sent_int)

        device = self._get_device()
        char_batch = self.char_emb(torch.tensor(char_batch, device=device))
        case_batch = self.case_emb(torch.tensor(case_batch, device=device))

        char_emb = torch.cat([char_batch, case_batch], dim=2)
        char_batch = self.char_proj(char_emb)
        return char_batch, torch.tensor(word_batch, device=device)

# demo testing        
if __name__ == "__main__":
    lookup = Lookup("../../../scratch")
    
    from cube2.components.loaders.loaders import getSequenceDataLoader
    
    input_list = ["d:\\ud-treebanks-v2.4\\UD_Romanian-RRT\\ro_rrt-ud-train.conllu","d:\\ud-treebanks-v2.4\\UD_Romanian-Nonstandard\\ro_nonstandard-ud-train.conllu"]
    dataloader = getSequenceDataLoader(input_list, batch_size=2, lookup_object=lookup, num_workers=0, shuffle=False)
    batch = iter(dataloader).next()
    (lang_id_sequences_tensor, seq_lengths, word_sequences_tensor, char_sequences_tensor, symbol_sequences_tensor, seq_masks, char_seq_lengths, symbol_seq_lengths, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor) = batch
    print("Batch has {} instances".format(lang_id_sequences_tensor.size(0)))
    print("First sentence: {}".format(word_sequences_tensor[0]))
    print("First sentence's char_encoding: {}".format(char_sequences_tensor[0]))
    print("First sentence's symb_encoding: {}".format(symbol_sequences_tensor[0]))
    
    
    te = TextEncoder(lookup, input_dropout_prob=0.2, char_encoder_size=10, char_encoder_layers=3, char_input_embeddings_size=10, input_size=10, total_layers=4, aux_softmax_layer_index=2, output_size=10, mlp_output_size=10, dropout=0.3, ext_conditioning=None, device='cpu')
    
    output=te.forward(batch)
    
    
    
    