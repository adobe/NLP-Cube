import sys

sys.path.append('')
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from cube3.io_utils.objects import Document, Sentence, Token, Word
from cube3.io_utils.encodings import Encodings
from cube3.io_utils.config import TaggerConfig
import numpy as np

from cube3.networks.modules import WordGram


class Tagger(pl.LightningModule):
    def __init__(self, config: TaggerConfig, encodings: Encodings):
        super().__init__()
        self._config = config
        self._encodings = encodings

        self._word_net = WordGram(len(encodings.char2int), num_langs=encodings.num_langs + 1)

    def forward(self, x_sents, x_lang_sent, x_words_chars, x_words_case, x_lang_word, x_sent_len, x_word_len,
                x_sent_masks, x_word_masks):
        word_emb = self._word_net(x_words_chars, x_words_case, x_lang_word, x_word_masks, x_word_len)
        # print(word_emb)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x_sent, x_lang, x_word_chars, x_word_case, x_lang_word, x_sent_len, x_word_len, x_sent_masks, x_word_masks, y_upos, y_xpos, y_attrs = batch
        rezult = self.forward(x_sent, x_lang, x_word_chars, x_word_case, x_lang_word, x_sent_len, x_word_len,
                              x_sent_masks,
                              x_word_masks)

    def validation_step(self, batch, batch_idx):
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
        x_lang_sent = np.zeros((len(batch)), dtype=np.long)
        x_lang_word = []

        y_upos = np.zeros((x_sent.shape[0], x_sent.shape[1]), dtype=np.long)
        y_xpos = np.zeros((x_sent.shape[0], x_sent.shape[1]), dtype=np.long)
        y_attrs = np.zeros((x_sent.shape[0], x_sent.shape[1]), dtype=np.long)
        for iSent in range(len(batch)):
            sent = batch[iSent]
            x_lang_sent[iSent] = sent.lang_id + 1
            for iWord in range(len(sent.words)):
                word = sent.words[iWord]
                if word.upos in self._encodings.upos2int:
                    y_upos[iSent, iWord] = self._encodings.upos2int[word.upos]
                if word.xpos in self._encodings.xpos2int:
                    y_xpos[iSent, iWord] = self._encodings.xpos2int[word.xpos]
                if word.attrs in self._encodings.attrs2int:
                    y_attrs[iSent, iWord] = self._encodings.attrs2int[word.attrs]
                word = sent.words[iWord].word
                x_sent_masks[iSent, iWord] = 1
                w = word.lower()
                x_lang_word.append(sent.lang_id + 1)
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

        x_lang_word = np.array(x_lang_word)
        return torch.tensor(x_sent), \
               torch.tensor(x_lang_sent), \
               torch.tensor(x_word), \
               torch.tensor(x_word_case), \
               torch.tensor(x_lang_word), \
               torch.tensor(x_sent_len), \
               torch.tensor(x_word_len), \
               torch.tensor(x_sent_masks), \
               torch.tensor(x_word_masks), \
               torch.tensor(y_upos), \
               torch.tensor(y_xpos), \
               torch.tensor(y_attrs)


if __name__ == '__main__':
    doc_train = Document(filename='corpus/ud-treebanks-v2.7/UD_Romanian-RRT/ro_rrt-ud-train.conllu')
    doc_dev = Document(filename='corpus/ud-treebanks-v2.7/UD_Romanian-RRT/ro_rrt-ud-dev.conllu')
    enc = Encodings()
    enc.compute(doc_train, None)

    trainset = TaggerDataset(doc_train)
    devset = TaggerDataset(doc_dev)

    collate = TaggerCollate(enc)
    train_loader = DataLoader(trainset, batch_size=32, collate_fn=collate.collate_fn)
    val_loader = DataLoader(devset, batch_size=32, collate_fn=collate.collate_fn)

    model = Tagger(config=None, encodings=enc)
    # training
    trainer = pl.Trainer(gpus=0, num_nodes=1, limit_train_batches=0.5, accelerator="ddp_cpu")
    trainer.fit(model, train_loader, val_loader)
