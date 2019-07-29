import os, sys, math
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
        MultiHeadAttention
        Inputs:
            encoder_outputs of size         [batch_size, enc_seq_len, enc_size]
            decoder_outputs_so_far of size  [batch_size, dec_seq_len, dec_size]
            mask (1s and 0s, default None), represents encoder padding [batch_size, enc_seq_len] for **keys**
        Output:
            context of size                 [batch_size, dec_seq_len, enc_size]
                where on each of the dec_seq_len positions there is an enc_size array that sums up the attention like:
                    for each i:
                        for all js: 
                            query_decoder_i * encoder_key_j, then multiply dot product with encoder_value_j
                        softmax then sum up all in an enc_size value (including multihead etc)
                            
        Init:
            d_model is basically the enc_size 
            num_heads must by a multiple
            custom_query_size is dec_size if used as an decored to encoder attention, otherwise the size of the encoder is used
    
        Note:
            use just as many decoder outputs as needed, masking is for encoder padding
    """
    def __init__(self, d_model, num_heads, dropout = 0.1, custom_query_size = None):
        # custom_query_size is for the decoder to encoder attention, to map the query to the d_model size, when the query size is different
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        
        if custom_query_size is None:
            self.q_linear = nn.Linear(d_model, d_model)
        else: 
            assert custom_query_size%num_heads==0, "MultiHeadAttention: custom_query_size({}) must be a multiple of num_heads({})!".format(custom_query_size, num_heads)
            self.q_linear = nn.Linear(custom_query_size, d_model)            
        
        assert d_model%num_heads==0, "MultiHeadAttention: d_model({}) must be a multiple of num_heads({})!".format(d_model, num_heads)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):        
        """
            Input: q*, k, v are [batch_size, seq_len, d_model] 
            Output:  [batch_size, 
        """
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads        
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_head)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_head)        
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_head)
        
        # transpose to get dimensions bs * h * sl * d_model       
        q = q.transpose(1,2)
        k = k.transpose(1,2)        
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = self._attention(q, k, v, self.d_head, mask, self.dropout)
        # scores is [batch_size, num_heads, dec_seq_len, enc_size/num_heads]   
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
        
    def _attention(self, q, k, v, d_k, mask=None, dropout=None):   
        # q is      [batch_size, num_heads, dec_seq_len, enc_size/num_heads]
        # k is      [batch_size, num_heads, enc_seq_len, enc_size/num_heads]
        # kt is     [batch_size, num_heads, enc_size, enc_seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        # scores is [batch_size, num_heads, dec_seq_len, enc_seq_len]   
        # scores represents, for the last 2 dims, for each query of the decoder i an array of scores 
        #   corresponding to the q_i * k_j where j is in (0, enc_seq_len)
        #   we want to mask all scores where the encoder is just padding 
        if mask is not None:
            # mask is [batch_size, enc_seq_len] of 1s and 0s            
            mask = mask.unsqueeze(1).unsqueeze(1)                         
            scores = scores.masked_fill(mask == 0, -1e9)                    
        scores = F.softmax(scores, dim=-1)        
        
        if dropout is not None:
            scores = dropout(scores)
        
        # scores is [batch_size, num_heads, dec_seq_len, enc_seq_len]        
        # v is      [batch_size, num_heads, enc_seq_len, enc_size/num_heads]        
        output = torch.matmul(scores, v)
        
        # output is [batch_size, num_heads, dec_seq_len, enc_size/num_heads]                
        return output    
        
if __name__ == "__main__":
    import numpy as np    
    import random
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # debug stuff:
    """scores = torch.ones((2,1,2,3), dtype=torch.float)
    mask = torch.tensor([ [1,1,0] , [0,0,0] ]) # [batch_size, enc_seq_len]
    print(scores)
    print(mask)    
    print("masking...")
    mask=mask.unsqueeze(1).unsqueeze(1)
    scores = scores.masked_fill(mask == 0, 0.)
    print(scores)
    print("-"*50)
    """
    
    # prep inputs
    batch_size = 3
    enc_seq_len = 5
    dec_seq_len = 4
    enc_size = 6    
    dec_size = 8
    
    encoder_outputs = torch.tensor(np.random.rand(batch_size, enc_seq_len, enc_size), dtype=torch.float)
    decoder_outputs_so_far = torch.tensor(np.random.rand(batch_size, dec_seq_len, dec_size), dtype=torch.float) 
    # mask is [batch_size, enc_seq_len] of 1s and 0s   
    mask = torch.tensor([[1,1,1,1,0],[1,1,1,0,0],[1,0,0,0,0]], dtype=torch.uint8)
    
    # prep layer
    #device = torch.device("cpu")
    multihead_att = MultiHeadAttention(d_model = enc_size, num_heads = 2, dropout = 0., custom_query_size = dec_size)
    
    # run
    context = multihead_att(q = decoder_outputs_so_far, k = encoder_outputs, v = encoder_outputs, mask = mask)
    
    
    out_batch_size = context.size()[0]
    out_seq_len = context.size()[1]
    out_size = context.size()[2]
    assert out_batch_size == batch_size, "Batch size failure"
    assert out_seq_len == dec_seq_len, "Seq len (dim 1) failure"
    assert out_size == enc_size, "Output size (dim 2) failure"
    print("Output is:")
    print(context)    
    print(context.size())    
        
    

            