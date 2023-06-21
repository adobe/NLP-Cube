import sys
import random
from typing import *

sys.path.append('')
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from cube.io_utils.objects import Document, Sentence, Token, Word
from cube.io_utils.encodings import Encodings

from collections import namedtuple


class DCWEDataset(Dataset):
    def __init__(self):
        self._examples = []

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        return self._examples[item]

    def load_language(self, filename: str, lang: str):
        f = open(filename, encoding='utf-8')
        parts = f.readline().strip().split(' ')
        num_examples = int(parts[0])
        vector_len = int(parts[1])
        for ii in range(num_examples):
            parts = f.readline().strip().split(' ')
            word = parts[0]
            vector = [float(pp) for pp in parts[1:]]
            self._examples.append([lang, word, vector])
        f.close()


class DCWECollate:
    encodings: Encodings
    examples: List[Any]

    def __init__(self, encodings: Encodings):
        self.encodings = encodings
        self._start = len(encodings.char2int)
        self._stop = len(encodings.char2int) + 1

    def collate_fn(self, examples):
        langs = []
        vectors = []
        words = []
        for example in examples:
            langs.append(example[0])
            words.append(example[1])
            vectors.append(example[2])

        max_word_len = max([len(word) for word in words]) + 2
        x_char = np.zeros((len(examples), max_word_len), dtype='long')
        x_case = np.zeros((len(examples), max_word_len), dtype='long')
        x_word_len = np.zeros((len(examples)), dtype='long')
        x_mask = np.ones((len(examples), 1))
        x_lang = np.ones((len(examples), 1))
        for ii in range(len(words)):
            word = words[ii]
            x_char[ii, 0] = self._start
            for jj in range(word):
                char = word[jj]
                ch_low = char.lower()
                if ch_low in self.encodings.char2int:
                    x_char[ii, jj + 1] = self.encodings.char2int[ch_low]
                else:
                    x_char[ii, jj + 1] = 1  # UNK
                if char.lower() == char.upper():
                    x_case[ii, jj + 1] = 1
                elif ch_low == char:
                    x_case[ii, jj + 1] = 2
                else:
                    x_case[ii, jj + 1] = 3

            x_char[len(word) + 1] = self._stop
            x_word_len[ii] = len(word)
            x_lang = self.encodings.lang2int[langs[ii]]

        collated = {'y_target': torch.tensor(np.array(vectors)),
                    'x_char': torch.tensor(x_char),
                    'x_case': torch.tensor(x_case),
                    'x_mask': torch.tensor(x_mask),
                    'x_lang': torch.tensor(x_lang),
                    'x_word_len': torch.tensor(x_word_len)}
