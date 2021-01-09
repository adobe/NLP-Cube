import sys

sys.path.append('')
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from cube3.io_utils.objects import Document, Sentence, Token, Word
from cube3.io_utils.encodings import Encodings
from cube3.io_utils.config import TaggerConfig
import numpy as np


class Tagger(pl.LightningModule):
    def __init__(self, config: TaggerConfig, encodings: Encodings):
        super().__init__()
        self._config = config
        self._encodings = encodings

    def forward(self, x_sents, x_words_chars, x_words_case, x_sent_len, x_word_len, x_sent_masks, x_word_masks):
        pass


class TaggerDataset(Dataset):
    def __init__(self, document: Document):
        self._document = document

    def __len__(self):
        return len(self._document.sentences)

    def __getitem__(self, item):
        return self._document.sentences[item]


class TaggerCollate:
    def __init__(self, encodings: Encodings):
        self._encodings = encodings

    def collate_fn(self, batch: [Sentence]):
        a_sent_len = [len(sent.words) for sent in batch]
        a_word_len = []
        for sent in batch:
            for word in sent.words:
                a_word_len.append(len(word.word))
        x_sent_len = np.array(a_sent_len, dtype=np.long)
        x_word_len = np.array(a_word_len, dtype=np.long)
        max_sent_len = np.max(x_sent_len)
        max_word_len = np.max(x_word_len)
        x_sent_masks = np.zeros((len(batch), max_sent_len), dtype=np.float)
        x_word_masks = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.float)

        x_sent = np.zeros((len(batch), max_sent_len), dtype=np.long)
        x_word = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.long)
        x_word_case = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.long)
        c_word = 0
        for iSent in range(len(batch)):
            sent = batch[iSent]
            for iWord in range(len(sent.words)):
                word = sent.words[iWord].word
                x_sent_masks[iSent, iWord] = 1
                w = word.lower()
                if w in self._encodings.word2int:
                    x_sent[iSent, iWord] = self._encodings.word2int[w]
                else:
                    x_sent[iSent, iWord] = self._encodings.word2int['<UNK>']

                for iChar in range(len(word)):
                    x_word_masks[c_word, iChar] = 1
                    ch = word[iChar]
                    if ch.lower() == ch.upper():  # symbol
                        x_word_case[c_word, iChar] = 1
                    elif ch.lower() != ch:  # upper
                        x_word_case[c_word, iChar] = 2
                    else:  # lower
                        x_word_case[c_word, iChar] = 3
                    ch = ch.lower()
                    if ch in self._encodings.char2int:
                        x_word[c_word, iChar] = self._encodings.char2int[ch]
                    else:
                        x_word[c_word, iChar] = self._encodings.char2int['<UNK>']
                c_word += 1
        return torch.tensor(x_sent), \
               torch.tensor(x_word), \
               torch.tensor(x_word_case), \
               torch.tensor(x_sent_len), \
               torch.tensor(x_word_len), \
               torch.tensor(x_sent_masks), \
               torch.tensor(x_word_masks)


if __name__ == '__main__':
    doc = Document(filename='corpus/ud-treebanks-v2.7/UD_French-GSD/fr_gsd-ud-dev.conllu')
    enc = Encodings()
    enc.compute(doc, None)

    collate = TaggerCollate(enc)
    batch = [doc.sentences[0], doc.sentences[1]]
    print(collate.collate_fn(batch))
