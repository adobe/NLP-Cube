import os, sys, json, random
sys.path.append("../../..")
import numpy as np
import torch
import torch.utils.data 

from cube2.components.lookup import Lookup
from cube2.components.datastructures import ConllEntry, Sequences


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