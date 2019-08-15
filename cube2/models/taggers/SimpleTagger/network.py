import sys
sys.path.append("../../../..")

import random
import numpy as np
import torch.nn as nn
import torch.utils.data

from cube2.components.interfaces import BaseTagger
from cube2.components.input.textencoder import TokenEncoder, LayeredRNN

class SimpleTagger(BaseTagger):
    def __init__(self, lookup):
        super(SimpleTagger, self).__init__()
        self.name = "SimpleTagger"
        
        self.lookup = lookup
        self.word_embedding_size = 100
        self.char_embedding_size = 100
        self.symbol_embedding_size = 16
        self.element_dropout = .2
        self.char_attention_num_heads = 4
        self.char_encoder_hidden_size = 256
        self.char_encoder_num_layers = 2
        self.char_encoder_rnn_dropout = .2
        self.encoder_rnn_hidden_size = 256
        self.encoder_rnn_hidden_layers = 5
        self.encoder_rnn_aux_layer_index = 3
        self.encoder_rnn_dropout = .4
        self.output_size = 200
        self.output_dropout = .3
        
        self.token_encoder = TokenEncoder(lookup, self.word_embedding_size, self.char_embedding_size, self.symbol_embedding_size, self.element_dropout,
            self.char_attention_num_heads, self.char_encoder_hidden_size, self.char_encoder_num_layers, self.char_encoder_rnn_dropout,
            self.encoder_rnn_hidden_size, self.encoder_rnn_hidden_layers, self.encoder_rnn_aux_layer_index, self.encoder_rnn_dropout,
            self.output_size, self.output_dropout, self.device)
        
        self.output_upos = nn.Linear(self.output_size, len(self.lookup.upos2int))
        self.output_xpos = nn.Linear(self.output_size, len(self.lookup.xpos2int))
        self.output_attrs = nn.Linear(self.output_size, len(self.lookup.attrs2int))

        #self.aux_mlp = nn.Sequential(nn.Linear(self.config.tagger_encoder_size * 2, self.output_size),
        #                             nn.Tanh(), nn.Dropout(p=self.config.tagger_mlp_dropout))
        #self.aux_output_upos = nn.Linear(self.output_size, len(self.lookup.upos2int))
        #self.aux_output_xpos = nn.Linear(self.output_size, len(self.lookup.xpos2int))
        #self.aux_output_attrs = nn.Linear(self.output_size, len(self.lookup.attrs2int))
        
        """self.embb = nn.Embedding(len(self.lookup.word2int), self.word_embedding_size, padding_idx=0) 
        
        self.lrnn = LayeredRNN(self.word_embedding_size, self.word_embedding_size, self.encoder_rnn_hidden_layers, self.encoder_rnn_dropout, self.encoder_rnn_hidden_size, self.output_size, self.encoder_rnn_dropout, pass_input_through_mlp = False, rnn_type=nn.LSTM, device=self.device)
        self.rnn = nn.LSTM(self.word_embedding_size,  self.encoder_rnn_hidden_size, num_layers=self.encoder_rnn_hidden_layers, dropout = self.encoder_rnn_dropout, bidirectional=True, batch_first=True)
        self.layer_dropout = nn.Dropout(self.encoder_rnn_dropout)
        self.output_upos = nn.Linear(2*self.encoder_rnn_hidden_size, len(self.lookup.upos2int))
        self.output_xpos = nn.Linear(2*self.encoder_rnn_hidden_size, len(self.lookup.xpos2int))
        self.output_attrs = nn.Linear(2*self.encoder_rnn_hidden_size, len(self.lookup.attrs2int))
        """
        self.to(self.device)

    def predict(self, input):
        """
            input is a list of sentences, where each sentence is a list of ConllEntry objects
            output is the same list, with UPOS, XPOS and ATTRS tags filled in
        """
        raise Exception("BaseTagger not implemented!")
        
    def save(self, folder):
        raise Exception("BaseTagger not implemented!")
        
    def load(self, folder):
        raise Exception("BaseTagger not implemented!")
    
    def forward(self, batch):
        (lang_id_sequences_tensor, word_sequences_tensor, word_seq_lengths, word_seq_masks, char_sequences_tensor, char_seq_lengths, char_seq_masks, symbol_sequences_tensor, symbol_seq_lengths, symbol_seq_masks, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor) = batch 
        
        emb, aux_hidden = self.token_encoder(lang_id_sequences_tensor, word_sequences_tensor, word_seq_lengths, word_seq_masks, char_sequences_tensor, char_seq_lengths, char_seq_masks, symbol_sequences_tensor, symbol_seq_lengths, symbol_seq_masks)
        
        #emb = self.embb(word_sequences_tensor)
        
        #emb, _ = self.lrnn(emb, seq_lengths)
        #emb = emb[-1]
        
        #pack_padded_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lengths, batch_first=True, enforce_sorted=False) # pack everything          
        #pack_padded_rnn_output, layer_hidden = self.rnn(pack_padded_rnn_input) # now run through the rnn layer            
        #layer_output, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_padded_rnn_output, batch_first=True) # undo the packing operation
        #emb = self.layer_dropout(layer_output) # dropout             
               
        s_upos = self.output_upos(emb)
        s_xpos = self.output_xpos(emb)
        s_attrs = self.output_attrs(emb)
        # aux_emb = torch.cat((hidden[self.config.aux_softmax_layer_index * 2, :, :],
        #                     hidden[0][self.config.aux_softmax_layer_index * 2 + 1, :, :]), dim=1)

        """
        aux_hid = self.aux_mlp(hidden)
        s_aux_upos = self.aux_output_upos(aux_hid)
        s_aux_xpos = self.aux_output_xpos(aux_hid)
        s_aux_attrs = self.aux_output_attrs(aux_hid)
        """
        
        return s_upos, s_xpos, s_attrs#, s_aux_upos, s_aux_xpos, s_aux_attrs

