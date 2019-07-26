import os, sys, json, random
import numpy as np

from lookup import Lookup
from datastructures import ConllEntry, Sequences
import torch
import torch.utils.data



def getSequenceDataLoader(input_list, batch_size, lookup_object, lookup_file=None, num_workers=None, shuffle=False, max_sequence_length=None, verbose=False):    
    """
        input_list is either:
                - a simple list containing the paths to the conllu filenames to be transformed. Language_ids will be generated incrementally
                - a list of tuples that on the first position has the conllu filename and the second a language_id        
    """
    #lookup_object = Lookup(lookup_folder, lookup_file) # if lookup_file is None, will load "lookup.json" from the folder
    #if num_workers is None: 
    #    num_workers = int(torch.get_num_threads()/2)
        
    return torch.utils.data.DataLoader(
                NLPCubeDataset(input_list, lookup_object, max_sequence_length=max_sequence_length, verbose=verbose),
                num_workers=int(torch.get_num_threads()/2) if num_workers is None else num_workers,
                batch_size=batch_size,
                collate_fn=_custom_collate_fn,
                shuffle=shuffle)
            
def _custom_collate_fn(insts): # padding function
    lang_id_sequences, word_sequences, char_sequences, symbol_sequences, upos_sequences, xpos_sequences, attrs_sequences = list(zip(*insts)) # insts is a list of batch_size Dataset instances    
        
    # determine lengths of sequences, pad and convert to tensor
    max_seq_len = max(len(inst) for inst in word_sequences) # determines max size for all sequences    
    seq_lengths = torch.tensor(list(map(len, word_sequences)), dtype=torch.long) # lenghts tensor   
    seq_masks = torch.tensor(np.array( [ [1] * len(inst) + [0] * (max_seq_len - len(inst)) for inst in word_sequences ] ), dtype=torch.long) # masks tensor
    
    lang_id_sequences_tensor = torch.tensor(lang_id_sequences, dtype=torch.long)
    word_sequences_tensor = torch.tensor(np.array( [ inst + [0] * (max_seq_len - len(inst)) for inst in word_sequences ] ), dtype=torch.long)
    upos_sequences_tensor = torch.tensor(np.array( [ inst + [0] * (max_seq_len - len(inst)) for inst in upos_sequences ] ), dtype=torch.long)
    xpos_sequences_tensor = torch.tensor(np.array( [ inst + [0] * (max_seq_len - len(inst)) for inst in xpos_sequences ] ), dtype=torch.long)
    attrs_sequences_tensor = torch.tensor(np.array( [ inst + [0] * (max_seq_len - len(inst)) for inst in attrs_sequences ] ), dtype=torch.long)
    
    max_char_seq_len = max([ len(char_list) for inst in char_sequences for char_list in inst ]) 
    np_char_sequences = []  
    np_char_seq_lengths = []        
    for char_instance in char_sequences:    
        np_char_sequence = []
        np_char_seq_length = []
        for char_list in char_instance:
            np_char_seq_length.append(len(char_list))
            np_char_sequence.append(char_list + [0]*(max_char_seq_len - len(char_list)))
        for _ in range(len(char_instance), max_seq_len): # pad the rest of elements of the sequence
            np_char_seq_length.append(0)
            np_char_sequence.append([0]*max_char_seq_len)                
        np_char_seq_lengths.append(np_char_seq_length)
        np_char_sequences.append(np_char_sequence)
    char_sequences_tensor = torch.tensor(np_char_sequences, dtype=torch.long)
    char_seq_lengths = torch.tensor(np_char_seq_lengths, dtype=torch.long)
    
    max_symb_seq_len = max([ len(symb_list) for inst in symbol_sequences for symb_list in inst ]) 
    np_symb_sequences = []  
    np_symb_seq_lengths = []        
    for symb_instance in symbol_sequences:    
        np_symb_sequence = []
        np_symb_seq_length = []
        for symb_list in symb_instance:
            np_symb_seq_length.append(len(symb_list))
            np_symb_sequence.append(symb_list + [0]*(max_symb_seq_len - len(symb_list)))
        for _ in range(len(symb_instance), max_seq_len): # pad the rest of elements of the sequence
            np_symb_seq_length.append(0)
            np_symb_sequence.append([0]*max_symb_seq_len)                
        np_symb_seq_lengths.append(np_symb_seq_length)
        np_symb_sequences.append(np_symb_sequence)
    symbol_sequences_tensor = torch.tensor(np_symb_sequences, dtype=torch.long)
    symbol_seq_lengths = torch.tensor(np_symb_seq_lengths, dtype=torch.long)    
    
    #print(char_sequences_tensor[0])
    #print(char_seq_lengths)
    # sort sequences
    #print(seq_lengths)
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    #print(perm_idx)
    #print(seq_lengths)  
    lang_id_sequences_tensor = lang_id_sequences_tensor[perm_idx]
    word_sequences_tensor = word_sequences_tensor[perm_idx]
    seq_masks = seq_masks[perm_idx]
    upos_sequences_tensor = upos_sequences_tensor[perm_idx]
    xpos_sequences_tensor = xpos_sequences_tensor[perm_idx]
    attrs_sequences_tensor = attrs_sequences_tensor[perm_idx]
    char_sequences_tensor = char_sequences_tensor[perm_idx]
    char_seq_lengths = char_seq_lengths[perm_idx]
    symbol_sequences_tensor = symbol_sequences_tensor[perm_idx]
    symbol_seq_lengths = symbol_seq_lengths[perm_idx]
    #print("--")
    #print(char_sequences_tensor[1])
    #print(char_seq_lengths)
    
    return lang_id_sequences_tensor, seq_lengths, word_sequences_tensor, char_sequences_tensor, symbol_sequences_tensor, seq_masks, char_seq_lengths, symbol_seq_lengths, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor
        
def getRawDataLoader():
    pass
        


class NLPCubeDataset(torch.utils.data.Dataset):
    """
        An NLPCubeDataset will contain sequences transformed to integers
    """
    def __init__(self, input_list, lookup_object, max_sequence_length=None, verbose=False):  
        """
            input_list is either:
                - a simple list containing the paths to the conllu filenames to be transformed. Language_ids will be generated incrementally
                - a list of tuples that on the first position has the conllu filename and the second a language_id            
                
            output: TODO
        """
        inputs = []
        if isinstance(input_list[0], str):
            for lang_id, filename in enumerate(input_list):
                inputs.append((filename, lang_id))
        else:
            inputs = input_list
            
        raw_sequences = []
        for filename, lang_id in inputs:
            dataset = Sequences(filename, lang_id, verbose)            
            raw_sequences += dataset.sequences
        if verbose:
            print("Read {} sequences, now processing ... ".format(len(raw_sequences)))
        
        self.lang_id_sequences = []
        self.word_sequences = [] 
        self.char_sequences = [] 
        self.symbol_sequences = [] 
        self.upos_sequences = []
        self.xpos_sequences = []
        self.attrs_sequences = []
        
        for sequence, lang_id in raw_sequences: # for each sequence, encode it as ints (except words -> tuple)
            word_sequence = []
            char_sequence = []
            symbol_sequence = []
            upos_sequence = []
            xpos_sequence = []
            attrs_sequence = []
            
            for entry in sequence:
                word_encoding, char_encoding, symbol_encoding = lookup_object.encode_word(entry.word)
                word_sequence.append(word_encoding)
                char_sequence.append(char_encoding)
                symbol_sequence.append(symbol_encoding)
                upos_sequence.append(lookup_object.encode_upos(entry.upos))
                xpos_sequence.append(lookup_object.encode_xpos(entry.xpos))
                attrs_sequence.append(lookup_object.encode_attr(entry.attrs))
                
            self.lang_id_sequences.append(lang_id)            
            self.word_sequences.append(word_sequence)
            self.char_sequences.append(char_sequence)
            self.symbol_sequences.append(symbol_sequence)
            self.upos_sequences.append(upos_sequence)
            self.xpos_sequences.append(xpos_sequence)
            self.attrs_sequences.append(attrs_sequence)
        
        if verbose:
            print("Done.")
            
    def __len__(self):
        return len(self.word_sequences)

    def __getitem__(self, idx):        
        return self.lang_id_sequences[idx], self.word_sequences[idx], self.char_sequences[idx], self.symbol_sequences[idx], self.upos_sequences[idx], self.xpos_sequences[idx], self.attrs_sequences[idx]

if __name__ == "__main__": # test functionality
    input_list = [("e:\\ud-treebanks-v2.4\\UD_Romanian-RRT\\ro_rrt-ud-train.conllu",3),("e:\\ud-treebanks-v2.4\\UD_Romanian-Nonstandard\\ro_nonstandard-ud-train.conllu",4)]    
    lookup_object = Lookup(".", lookup_file=None) # a lookup.json should already be present in this folder for testing purposes!
    loader = getSequenceDataLoader(input_list, batch_size = 3, lookup_object = lookup_object, num_workers = 0)
    batch = iter(loader).next()
    (lang_id_sequences_tensor, seq_lengths, word_sequences_tensor, char_sequences_tensor, symbol_sequences_tensor, seq_masks, char_seq_lengths, symbol_seq_lengths, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor) = batch
    print("Batch has {} instances".format(lang_id_sequences_tensor.size(0)))
    print("First sentence: {}".format(word_sequences_tensor[0]))
    print("First sentence's char_encoding: {}".format(char_sequences_tensor[0]))
    print("First sentence's symb_encoding: {}".format(symbol_sequences_tensor[0]))
    