import sys, os
sys.path.append("../../..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from cube2.components.lookup import Lookup
from cube2.components.attention.MultiHeadAttention import MultiHeadAttention

class LayeredRNN(nn.Module):
    def __init__(self, input_embedding_size, rnn_input_size, rnn_num_layers, rnn_dropout, rnn_hidden_size, output_size, output_dropout, pass_input_through_mlp = True, rnn_type=nn.GRU):
        """
            TODO docs
        """
        super().__init__()    
        self.input_embedding_size = input_embedding_size # this is the initial embedding size of an element; if pass_input_through_mlp is False, it should be equal to rnn_input_size
        self.rnn_input_size = rnn_input_size # this is the embedding size that will be passed to the RNN 
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout = rnn_dropout
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = output_size
        self.output_dropout = output_dropout
        self.pass_input_through_mlp = pass_input_through_mlp 
        self.rnn_type = rnn_type
        
        if self.pass_input_through_mlp == True:
            self.pre_rnn_mlp_layer = nn.Sequential(nn.Linear(input_embedding_size, rnn_input_size), nn.Tanh())
        else:
            if input_embedding_size != rnn_input_size:
                raise Exception("ERROR: input_embedding_size should be equal to rnn_input_size if no initial MLP will be applied!")
                
        self.rnns = []
        layer_input_size = rnn_input_size
        for i in range(rnn_num_layers):            
            self.rnns.append(rnn_type(layer_input_size, rnn_hidden_size, num_layers=1, bidirectional=True, batch_first=True))
            layer_input_size = rnn_hidden_size * 2
        self.rnn_layer_dropout = nn.Dropout(rnn_dropout)
        
        self.fc = nn.Linear(rnn_hidden_size * 2, output_size)
        self.dropout = nn.Dropout(output_dropout)

    def forward(self, input, input_lengths):        
        """
            input:          [batch_size, seq_len, encoding_size] 
            input_lengths:  [batch_size, seq_len] 
        """
        if self.pass_input_through_mlp == True:
            layer_input = self.pre_rnn_mlp_layer(input)
        else:
            layer_input = input
        # input is now [batch_size, seq_len, rnn_input_size] 
        outputs = []
        hiddens = []        
        for i in range(self.rnn_num_layers): 
            pack_padded_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(layer_input, input_lengths, batch_first=True, enforce_sorted=False) # pack everything            
            pack_padded_rnn_output, layer_hidden = self.rnns[i](pack_padded_rnn_input) # now run through the rnn layer            
            layer_output, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_padded_rnn_output, batch_first=True) # undo the packing operation
            
            layer_output = self.rnn_layer_dropout(layer_output) # dropout             
            layer_input = layer_output
            
            if isinstance(layer_hidden, list) or isinstance(layer_hidden, tuple):  # we have an LSTM
                layer_hidden = layer_hidden[1] # this is the cell state !!, [0] is hidden, [1] is the cell_state
                
            outputs.append(layer_output) # each layer_output is [batch_size, seq_len, num_directions * hidden_size]
            # each layer_hidden is [num_layers * num_directions, batch_size, hidden_size]
            #fc_hidden = torch.tanh(self.fc(torch.cat((layer_hidden[-2, :, :], layer_hidden[-1, :, :]), dim=1)))
            hiddens.append(layer_hidden) 
      
        # TODO output_size removed, fc-hidden??
       
        return outputs, hiddens

class SimpleSelfAttentionEncoder(nn.Module):
    def __init__(self, input_size, attention_num_heads, encoder_hidden_size, encoder_num_layers, encoder_rnn_dropout, output_size, encoder_dropout, encoder_rnn_type=nn.GRU):
        super(SimpleSelfAttentionEncoder, self).__init__()
        self.input_size = input_size
        self.attention_num_heads = attention_num_heads
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.encoder_rnn_dropout = encoder_rnn_dropout
        self.output_size = output_size
        self.encoder_dropout = encoder_dropout
        self.encoder_rnn_type = encoder_rnn_type
        
        self.rnn = LayeredRNN(input_size, input_size, encoder_num_layers, encoder_rnn_dropout, encoder_hidden_size, output_size, encoder_rnn_dropout, pass_input_through_mlp = True, rnn_type=encoder_rnn_type)
        self.mhattention = MultiHeadAttention(encoder_hidden_size*2, attention_num_heads, dropout = encoder_dropout, custom_query_size = None)
        self.output_mlp = nn.Linear(encoder_hidden_size*2 + encoder_hidden_size*2, output_size) # from encoder output and from attention (same size as encoder output)
        
    def forward(self, input, input_lengths):
        """
            input:          [batch_size, seq_len, encoding_size] 
            input_lengths:  [batch_size, seq_len] 
        """
        rnn_outputs, rnn_hiddens = self.rnn(input, input_lengths)
        # rnn_outputs is a list of [batch_size, seq_len, num_directions(2) * hidden_size]
        # rnn_hiddens is a list of [num_layers * num_directions(2), batch_size, hidden_size]
        
        # run multihead attention
        # input q, k, v are [batch_size, seq_len, encoder_hidden_size*2]         
        context = self.mhattention(q = rnn_outputs[-1], k = rnn_outputs[-1], v = rnn_outputs[-1], mask = None)
        # context is [batch_size, seq_len, encoding_size]
        
        # run mlp         
        mlp_input = torch.cat([rnn_outputs[-1], context], dim=2)
        
        output = self.output_mlp(mlp_input)
        return output

class TokenEncoder(nn.Module):
    def __init__(self, lookup, word_embedding_size, char_embedding_size, symbol_embedding_size, element_dropout,         
        char_attention_num_heads, char_encoder_hidden_size, char_encoder_num_layers, char_encoder_rnn_dropout, 
        #char_projection_size, # char+symbol get projected into this size
        encoder_rnn_hidden_size, encoder_rnn_hidden_layers, encoder_rnn_aux_layer_index, encoder_rnn_dropout):
        
        super(TokenEncoder, self).__init__()
         
        self.lookup = lookup
        self.word_embedding_size = word_embedding_size
        self.char_embedding_size = char_embedding_size
        self.symbol_embedding_size = symbol_embedding_size
        self.element_dropout = element_dropout
        
        self.word_embedding = nn.Embedding(len(self.lookup.word2int), word_embedding_size, padding_idx=0) 
        self.char_embedding = nn.Embedding(len(self.lookup.char2int), char_embedding_size, padding_idx=0) 
        self.symb_embedding = nn.Embedding(len(self.lookup.symbol2int), symbol_embedding_size, padding_idx=0) 
        
        assert (char_embedding_size+symbol_embedding_size)%attention_num_heads == 0, "char_embedding_size+symbol_embedding_size must be divisible with attention_num_heads"
        self.character_network = SimpleSelfAttentionEncoder(input_size = char_embedding_size+symbol_embedding_size, attention_num_heads = char_attention_num_heads, encoder_hidden_size = char_encoder_hidden_size, encoder_num_layers = char_encoder_num_layers, encoder_rnn_dropout = char_encoder_rnn_dropout, output_size = char_encoder_hidden_size, encoder_dropout = char_encoder_rnn_dropout, encoder_rnn_type=nn.LSTM)        
        #self.character_network_projection = nn.Linear(self.character_network.output_size, char_projection_size)
        
    def forward(self, lang_ids, input_words, input_words_length, input_chars, input_chars_lengths, input_symbols, input_symbols_lengths):
        assert len(input_words[0]) == len(input_chars[0]), "Sequence lengths invalid!"
        assert len(input_words[0]) == len(input_symbols[0]), "Sequence lengths invalid!"
        
        # first, compute character encodings
        batch_size = lang_ids.size(0)
        seq_len = input_chars.size(1)
        char_seq_len = input_chars.size(2)
        symb_seq_len = input_symbols.size(2)
        
        print("Batch size {}, seq_len {}, char_len {}, symb_len {}".format(batch_size, seq_len, char_seq_len, symb_seq_len))
        
        word_emb = self.word_embedding(input_words)
        char_emb = self.char_embedding(input_chars)        
        symb_emb = self.symb_embedding(input_symbols)        
        char_emb = torch.cat([char_emb, symb_emb], dim=3) # this is now the concatenation of char and symbol embeddings        
        # char_emb [batch_size, seq_len, char_len, char_embedding_size+symbol_embedding_size]
        
        print("For each sentence:")
        print(input_chars)
        print(input_chars_lengths)
        char_encodings_list = []
        for index in range(batch_size):
            word_count = input_words_length[index]
            #print("\nInstance "+str(index)+", # of words {}".format(word_count))
            #print(char_emb[index].size())
            #print(input_chars_lengths[index].size())
            #print(input_chars_lengths[index])
            instance_char_emb = char_emb[index][:word_count]
            #print(" after cut:")
            #print(instance_char_emb.size())
            instance_char_lengths = input_chars_lengths[index][:word_count]
            #print(instance_char_lengths)
            # instance_char_emb [word_count, char_seq_len, char_embedding_size+symbol_embedding_size]
            
            char_encodings = self.character_network(instance_char_emb, instance_char_lengths) # [word_count, max_char_len(variable), char_encoder_hidden_size]            
            #print(char_encodings.size())
            # we want to have a single encoding per word, so we sum all the characters together
            char_encodings = torch.sum(char_encodings, dim=1) # [word_count, char_encoder_hidden_size]
            #print(char_encodings.size())
            char_encodings_list.append(char_encodings)
        
        # word_emb [batch_size, seq_len, word_embedding_size]
        encoder_inputs = torch.zeros((batch_size, seq_len, self.word_embedding_size+self.character_network.output_size), device = word_emb.device)
        
        # create masks
        for index in range(batch_size):
            word_drop, char_drop = random.random(), random.random()
            if word_drop >= prob and p2 < prob:
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
    

        #char_network_output = self.character_network(
         #char_encoder_hidden_size
        
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
        
    def _compute_masks():
        pass

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
        """
            x is a batch containing padded sequences of words(list of ints), characters(list of list of ints) and symbol encodings(list of list of ints)
        """
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
    # test LayeredRNN
    
    batch_size = 1
    seq_len = 5    
    input_embedding_size = 2        
    rnn_num_layers = 3
    rnn_hidden_size = 4
    input = torch.tensor(np.random.rand(batch_size, seq_len, input_embedding_size), dtype=torch.float)    
    input_lengths = torch.tensor([seq_len], dtype=torch.long)    
    # mask is [batch_size, enc_seq_len] of 1s and 0s   
    # mask = torch.tensor([[1,1,1,1,0],[1,1,1,0,0],[1,0,0,0,0]], dtype=torch.uint8)
    
    # ####################################################################################
    print("\n\n\nTest LayeredRNN")
    lrnn = LayeredRNN(input_embedding_size, input_embedding_size, rnn_num_layers=rnn_num_layers, rnn_dropout=0., rnn_hidden_size=rnn_hidden_size, output_size=rnn_hidden_size, output_dropout=0., pass_input_through_mlp = False, rnn_type=nn.LSTM)
    outputs, hiddens = lrnn(input, input_lengths)
    print("Last layer output: {}".format(outputs[-1].size()))
    print("Last layer hidden: {}".format(hiddens[-1].size()))
    
    # ####################################################################################
    
    print("\n\n\nTest SimpleSelfAttentionEncoder")
    output_size = 6
    attention_num_heads = 2
    sate = SimpleSelfAttentionEncoder(input_embedding_size, attention_num_heads, rnn_hidden_size, rnn_num_layers, encoder_rnn_dropout=0., output_size=output_size, encoder_dropout=0., encoder_rnn_type=nn.LSTM)
    output = sate(input, input_lengths)
    print("Output: {}".format(output.size()))
    
    # ####################################################################################
    
    print("\n\n\nTest SelfAttentionTokenEncoder")
    
    lookup = Lookup("../../../scratch")
    
    from cube2.components.loaders.loaders import getSequenceDataLoader
    
    input_list = ["e:\\ud-treebanks-v2.4\\UD_Romanian-RRT\\ro_rrt-ud-train.conllu","e:\\ud-treebanks-v2.4\\UD_Romanian-Nonstandard\\ro_nonstandard-ud-train.conllu"]
    dataloader = getSequenceDataLoader(input_list, batch_size=2, lookup_object=lookup, num_workers=0, shuffle=False)
    batch = iter(dataloader).next()
    (lang_id_sequences_tensor, seq_lengths, word_sequences_tensor, char_sequences_tensor, symbol_sequences_tensor, seq_masks, char_seq_lengths, symbol_seq_lengths, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor) = batch
    #print("Batch has {} instances".format(lang_id_sequences_tensor.size(0)))
    #print("First sentence: {}".format(word_sequences_tensor[0]))
    #print("First sentence's char_encoding: {}".format(char_sequences_tensor[0]))
    #print("First sentence's symb_encoding: {}".format(symbol_sequences_tensor[0]))
    
    word_embedding_size, char_embedding_size, symbol_embedding_size, element_dropout = 100,112,16,0.4
    char_attention_num_heads, char_encoder_hidden_size, char_encoder_num_layers, char_encoder_rnn_dropout = 4, 64, 2, .33
    encoder_rnn_hidden_size, encoder_rnn_hidden_layers, encoder_rnn_aux_layer_index, encoder_rnn_dropout = 256, 5, 3, .33
    
    te = TokenEncoder(lookup, word_embedding_size, char_embedding_size, symbol_embedding_size, element_dropout
        char_attention_num_heads, char_encoder_hidden_size, char_encoder_num_layers, char_encoder_rnn_dropout,
        encoder_rnn_hidden_size, encoder_rnn_hidden_layers, encoder_rnn_aux_layer_index, encoder_rnn_dropout)
    
    output = te(lang_id_sequences_tensor, word_sequences_tensor, seq_lengths, char_sequences_tensor, char_seq_lengths, symbol_sequences_tensor, symbol_seq_lengths)
    print("Output: {}".format(output.size()))
    
    #te = TextEncoder(lookup, input_dropout_prob=0.2, char_encoder_size=10, char_encoder_layers=3, char_input_embeddings_size=10, input_size=10, total_layers=4, aux_softmax_layer_index=2, output_size=10, mlp_output_size=10, dropout=0.3, ext_conditioning=None, device='cpu')    
    #output=te.forward(batch)
    
    
    
    