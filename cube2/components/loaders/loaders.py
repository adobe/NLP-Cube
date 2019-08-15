import os, sys, json, random
sys.path.append("../../..")
import numpy as np

from cube2.components.lookup import Lookup
from cube2.components.loaders.datasets import NLPCubeDataset
from cube2.components.datastructures import ConllEntry, Sequences
import torch
import torch.utils.data



def getSequenceDataLoader(input_list, batch_size, lookup_object, num_workers=None, shuffle=False, max_sequence_length=None, verbose=False):    
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
    word_seq_lengths = torch.tensor(list(map(len, word_sequences)), dtype=torch.long) # lenghts tensor   
    word_seq_masks = torch.tensor(np.array( [ [1] * len(inst) + [0] * (max_seq_len - len(inst)) for inst in word_sequences ] ), dtype=torch.uint8) # masks tensor for words
    
    lang_id_sequences_tensor = torch.tensor(lang_id_sequences, dtype=torch.long)
    word_sequences_tensor = torch.tensor(np.array( [ inst + [0] * (max_seq_len - len(inst)) for inst in word_sequences ] ), dtype=torch.long)
    upos_sequences_tensor = torch.tensor(np.array( [ inst + [0] * (max_seq_len - len(inst)) for inst in upos_sequences ] ), dtype=torch.long)
    xpos_sequences_tensor = torch.tensor(np.array( [ inst + [0] * (max_seq_len - len(inst)) for inst in xpos_sequences ] ), dtype=torch.long)
    attrs_sequences_tensor = torch.tensor(np.array( [ inst + [0] * (max_seq_len - len(inst)) for inst in attrs_sequences ] ), dtype=torch.long)
    
    max_char_seq_len = max([ len(char_list) for inst in char_sequences for char_list in inst ]) 
    np_char_sequences = []  
    np_char_seq_lengths = []   
    np_char_masks = []
    for char_instance in char_sequences:    
        np_char_sequence = []
        np_char_seq_length = []
        np_char_mask = []
        for char_list in char_instance:
            np_char_seq_length.append(len(char_list))
            np_char_sequence.append(char_list + [0]*(max_char_seq_len - len(char_list)))
            np_char_mask.append([1]*len(char_list) + [0]*(max_char_seq_len - len(char_list)))
        for _ in range(len(char_instance), max_seq_len): # pad the rest of elements of the sequence with zeroes
            np_char_seq_length.append(0)
            np_char_sequence.append([0]*max_char_seq_len) 
            np_char_mask.append([0]*max_char_seq_len)
        np_char_seq_lengths.append(np_char_seq_length)
        np_char_sequences.append(np_char_sequence)
        np_char_masks.append(np_char_mask)
    char_seq_lengths = torch.tensor(np_char_seq_lengths, dtype=torch.long)
    char_sequences_tensor = torch.tensor(np_char_sequences, dtype=torch.long)    
    char_seq_masks = torch.tensor(np_char_masks, dtype=torch.uint8) # ByteTensor
    
    max_symb_seq_len = max([ len(symb_list) for inst in symbol_sequences for symb_list in inst ]) 
    np_symb_sequences = []  
    np_symb_seq_lengths = []      
    np_symb_masks = []    
    for symb_instance in symbol_sequences:    
        np_symb_sequence = []
        np_symb_seq_length = []
        np_symb_mask = []
        for symb_list in symb_instance:
            np_symb_seq_length.append(len(symb_list))
            np_symb_sequence.append(symb_list + [0]*(max_symb_seq_len - len(symb_list)))
            np_symb_mask.append([1]*len(symb_list) + [0]*(max_symb_seq_len - len(symb_list)))
        for _ in range(len(symb_instance), max_seq_len): # pad the rest of elements of the sequence
            np_symb_seq_length.append(0)
            np_symb_sequence.append([0]*max_symb_seq_len)  
            np_symb_mask.append([0]*max_symb_seq_len)            
        np_symb_seq_lengths.append(np_symb_seq_length)
        np_symb_sequences.append(np_symb_sequence)
        np_symb_masks.append(np_symb_mask)
    symbol_seq_lengths = torch.tensor(np_symb_seq_lengths, dtype=torch.long)  
    symbol_sequences_tensor = torch.tensor(np_symb_sequences, dtype=torch.long)      
    symbol_seq_masks = torch.tensor(np_symb_masks, dtype=torch.uint8) # ByteTensor
    #print(char_sequences_tensor[0])
    #print(char_seq_lengths)
    # sort sequences
    #print(word_seq_lengths)
    word_seq_lengths, perm_idx = word_seq_lengths.sort(0, descending=True)
    #print(perm_idx)
    #print(word_seq_lengths)      
    lang_id_sequences_tensor = lang_id_sequences_tensor[perm_idx]
    word_sequences_tensor = word_sequences_tensor[perm_idx]
    word_seq_masks = word_seq_masks[perm_idx]
    
    char_sequences_tensor = char_sequences_tensor[perm_idx]
    char_seq_lengths = char_seq_lengths[perm_idx]
    char_seq_masks = char_seq_masks[perm_idx]
    
    symbol_sequences_tensor = symbol_sequences_tensor[perm_idx]
    symbol_seq_lengths = symbol_seq_lengths[perm_idx]
    symbol_seq_masks = symbol_seq_masks[perm_idx]
    
    upos_sequences_tensor = upos_sequences_tensor[perm_idx]
    xpos_sequences_tensor = xpos_sequences_tensor[perm_idx]
    attrs_sequences_tensor = attrs_sequences_tensor[perm_idx]
    
    
    #print("--")
    #print(char_sequences_tensor[1])
    #print(char_seq_lengths)
    
    return lang_id_sequences_tensor, word_sequences_tensor, word_seq_lengths, word_seq_masks, char_sequences_tensor, char_seq_lengths, char_seq_masks, symbol_sequences_tensor, symbol_seq_lengths, symbol_seq_masks, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor
        
def getRawDataLoader():
    pass
        

if __name__ == "__main__": # test functionality
    input_list = [("e:\\ud-treebanks-v2.4\\UD_Romanian-RRT\\ro_rrt-ud-train.conllu",3),("e:\\ud-treebanks-v2.4\\UD_Romanian-Nonstandard\\ro_nonstandard-ud-train.conllu",4)]    
    lookup_object = Lookup(".", lookup_file=None) # a lookup.json should already be present in this folder for testing purposes!
    loader = getSequenceDataLoader(input_list, batch_size = 3, lookup_object = lookup_object, num_workers = 0)
    batch = iter(loader).next()
    (lang_id_sequences_tensor, word_sequences_tensor, word_seq_lengths, word_seq_masks, char_sequences_tensor, char_seq_lengths, char_seq_masks, symbol_sequences_tensor, symbol_seq_lengths, symbol_seq_masks, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor) = batch
    print("Batch has {} instances".format(lang_id_sequences_tensor.size(0)))
    print("First sentence: {}".format(word_sequences_tensor[0]))
    print("First sentence's char_encoding: {}".format(char_sequences_tensor[0]))
    print("First sentence's symb_encoding: {}".format(symbol_sequences_tensor[0]))
    