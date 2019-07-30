import sys, os, json, collections, string
sys.path.insert(0, '../..')

from itertools import dropwhile
from cube2.components.datastructures import ConllEntry, Sequences


class Lookup ():
    def __init__(self, lookup_folder=None, lookup_file=None):
        self.symbol2int = {"<PAD>":0, "<UNK>":1, "UPPER":2, "LOWER":3, "SYMBOL":4} # this is constant, does not change
        self.int2symbol = self._inverse_dictionary(self.symbol2int) 
        
        if lookup_folder is None:
            self.word2int = {"<PAD>":0, "<UNK>":1}
            self.int2word = {}
            
            self.char2int = {"<PAD>":0, "<UNK>":1}
            self.int2char = {}
            
            self.upos2int = {"<PAD>":0, "<UNK>":1}
            self.int2upos = {}
            
            self.xpos2int = {"<PAD>":0, "<UNK>":1}
            self.int2xpos = {}
            
            self.attrs2int = {"<PAD>":0, "<UNK>":1}
            self.int2attrs = {}        
        else: 
            self.load(lookup_folder, lookup_file)
    
    def encode_word(self, word): # word is a string          
        word_lowercased = word.lower()                
        word_encoding = self.word2int[word_lowercased] if word_lowercased in self.word2int else self.word2int["<UNK>"]        
        char_encoding = []
        symbol_encoding = []        
        for char_lowercased, char_unchanged in zip(word_lowercased, word):            
            char_encoding.append(self.char2int[char_lowercased] if char_lowercased in self.char2int else self.char2int["<UNK>"])            
            if char_lowercased in string.punctuation:   
                symbol_encoding.append(self.symbol2int["SYMBOL"]) # this is punctuation
            elif char_lowercased != char_unchanged:
                symbol_encoding.append(self.symbol2int["UPPER"]) # char is uppercase
            else:
                symbol_encoding.append(self.symbol2int["LOWER"]) # char is lowercase
        return (word_encoding, char_encoding, symbol_encoding)
    
    def encode_upos(self, upos): # upos is a string        
        return self.upos2int[upos] if upos in self.upos2int else self.upos2int["<UNK>"]
    
    def encode_xpos(self, xpos): # xpos is a string        
        return self.xpos2int[xpos] if xpos in self.xpos2int else self.xpos2int["<UNK>"]
    
    def encode_attr(self, attr): # attr is a string        
        return self.attrs2int[attr] if attr in self.attrs2int else self.attrs2int["<UNK>"]
    
    def _inverse_dictionary(self, d):
        reverse_d = {}
        for key in d:
            value = d[key]
            reverse_d[str(value)] = str(key)
        return reverse_d
        
    def load(self, folder, filename = None):
        save_path = os.path.join(folder, "lookup.json") if filename is None else os.path.join(folder, filename)
        with open(save_path, "r", encoding="utf8") as file:
            self.__dict__ = json.load(file)
        
    def save(self, folder, filename = None):
        save_path = os.path.join(folder, "lookup.json") if filename is None else os.path.join(folder, filename)
        with open(save_path, "w", encoding="utf8") as file:
            json.dump(self.__dict__, file, indent=2)
        
    def __repr__(self):
        return "Lookup contains {} words, {} chars, {} UPOSes, {} XPOSes, {} attrs".format(len(self.word2int), len(self.char2int), len(self.upos2int), len(self.xpos2int), len(self.attrs2int))
        
def createLookup(conll_files, minimum_word_frequency_cutoff=7, minimum_char_frequency_cutoff=5, verbose=False):
    """
        This function reads a conll_file and creates word2index, char2index and symbol2index dictionaries
        conll_files_list is a list of files
    """
    if verbose:
        print("Creating Lookup object with word cutoff at {} and char cutoff at {}...".format(minimum_word_frequency_cutoff, minimum_char_frequency_cutoff))
    lookup = Lookup() # empty object
    for lang_id, conll_file in enumerate(conll_files):
        if verbose:
            print("Processing file: [{}] with lang_id = {}".format(conll_file, lang_id))
        
        dataset = Sequences(conll_file, lang_id)
        
        word_frequency = collections.Counter()
        char_frequency = collections.Counter()
        
        for seq, lang_id in dataset.sequences:    
            word_frequency.update([entry.word.lower() for entry in seq])
            for entry in seq:
                char_frequency.update(entry.word.lower())
                
        # drop less frequent words and create word2int
        if verbose:
            print("Total words: {}".format(len(word_frequency)))        
        for key, count in dropwhile(lambda key_count: key_count[1] >= minimum_word_frequency_cutoff, word_frequency.most_common()):
            del word_frequency[key]   
        for word, count in word_frequency.most_common():
            if word not in lookup.word2int:
                lookup.word2int[word] = len(lookup.word2int)
        if verbose:
            print("\tAfter cutoff remaining words: {}".format(len(word_frequency)))
        
        # drop less frequent chars and create char2int
        if verbose:
            print("Total chars: {}".format(len(char_frequency)))            
        for key, count in dropwhile(lambda key_count: key_count[1] >= minimum_char_frequency_cutoff, char_frequency.most_common()):
            del char_frequency[key]    
        for character, count in char_frequency.most_common():
            if character not in lookup.char2int:
                lookup.char2int[character] = len(lookup.char2int)    
        for digit in "0123456789": # force add digits
            if digit not in lookup.char2int:
                lookup.char2int[digit] = len(lookup.char2int)            
        if verbose:
            print("\tAfter cutoff remaining chars: {}".format(len(char_frequency)))
        
        # create other indexes
        for seq, lang_id in dataset.sequences:    
            for entry in seq:
                if entry.upos not in lookup.upos2int:
                    lookup.upos2int[entry.upos] = len(lookup.upos2int)
                if entry.xpos not in lookup.xpos2int:
                    lookup.xpos2int[entry.xpos] = len(lookup.xpos2int)
                if entry.attrs not in lookup.attrs2int:
                    lookup.attrs2int[entry.attrs] = len(lookup.attrs2int)    
                
    # create reverse indices
    lookup.int2word = lookup._inverse_dictionary(lookup.word2int)
    lookup.int2char = lookup._inverse_dictionary(lookup.char2int)
    lookup.int2upos = lookup._inverse_dictionary(lookup.upos2int)
    lookup.int2xpos = lookup._inverse_dictionary(lookup.xpos2int)
    lookup.int2attrs = lookup._inverse_dictionary(lookup.attrs2int)
    
    if verbose:
        print(lookup)
        
    return lookup

# demo testing        
if __name__ == "__main__":
    lookup = createLookup(["e:\\ud-treebanks-v2.4\\UD_Romanian-RRT\\ro_rrt-ud-train.conllu","e:\\ud-treebanks-v2.4\\UD_Romanian-Nonstandard\\ro_nonstandard-ud-train.conllu"], verbose=True)      
    lookup.save("../../scratch")
    lookup.load("../../scratch")
    print(lookup)
    new_lookup = Lookup("../../scratch")
    print(new_lookup.encode_word("Mere!"))
    print(new_lookup.encode_word("numai"))