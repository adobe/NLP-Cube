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
from cube3.networks.modules import ConvNorm, LinearNorm

from cube3.networks.modules import WordGram


class Tagger(pl.LightningModule):
    def __init__(self, config: TaggerConfig, encodings: Encodings):
        super().__init__()
        self._config = config
        self._encodings = encodings

        self._word_net = WordGram(len(encodings.char2int), num_langs=encodings.num_langs + 1,
                                  num_filters=config.char_filter_size, char_emb_size=config.char_emb_size,
                                  lang_emb_size=config.lang_emb_size, num_layers=config.char_layers)
        self._zero_emb = nn.Embedding(1, config.char_filter_size // 2)

        conv_layers = []
        cs_inp = config.char_filter_size // 2 + config.lang_emb_size + config.word_emb_size
        NUM_FILTERS = config.cnn_filter
        for _ in range(config.cnn_layers):
            conv_layer = nn.Sequential(
                ConvNorm(cs_inp,
                         NUM_FILTERS,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(NUM_FILTERS))
            conv_layers.append(conv_layer)
            cs_inp = NUM_FILTERS // 2 + config.lang_emb_size
        self._word_emb = nn.Embedding(len(encodings.word2int), config.word_emb_size, padding_idx=0)
        self._lang_emb = nn.Embedding(encodings.num_langs + 1, config.lang_emb_size, padding_idx=0)
        self._convs = nn.ModuleList(conv_layers)
        self._upos = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.upos2int))
        self._xpos = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.xpos2int))
        self._attrs = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.attrs2int))

    def forward(self, x_sents, x_lang_sent, x_words_chars, x_words_case, x_lang_word, x_sent_len, x_word_len,
                x_sent_masks, x_word_masks):
        char_emb_packed = self._word_net(x_words_chars, x_words_case, x_lang_word, x_word_masks, x_word_len)

        blist = []
        sl = x_sent_len.cpu().numpy()
        pos = 0
        for ii in range(x_sents.shape[0]):
            head = char_emb_packed[pos:pos + sl[ii], :]
            pos += sl[ii]
            tail = torch.zeros((x_sents.shape[1] - sl[ii], self._config.char_filter_size // 2),
                               device=self._get_device(), dtype=torch.float)
            sent_emb = torch.cat([head, tail], dim=0)
            blist.append(sent_emb.unsqueeze(0))
        char_emb = torch.cat(blist, dim=0).float()

        word_emb = self._word_emb(x_sents)
        lang_emb = self._lang_emb(x_lang_sent)
        lang_emb = lang_emb.unsqueeze(1).repeat(1, word_emb.shape[1], 1)

        x = torch.cat([word_emb, lang_emb, char_emb], dim=-1)
        x = x.permute(0, 2, 1)
        lang_emb = lang_emb.permute(0, 2, 1)
        half = self._config.cnn_filter // 2
        for conv in self._convs:
            conv_out = conv(x)
            tmp = torch.tanh(conv_out[:, :half, :]) * torch.sigmoid((conv_out[:, half:, :]))
            x = torch.dropout(tmp, 0.1, self.training)
            x = torch.cat([x, lang_emb], dim=1)
        x = x.permute(0, 2, 1)
        return self._upos(x), self._xpos(x), self._attrs(x)

    def _get_device(self):
        if self._lang_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._lang_emb.weight.device.type, str(self._lang_emb.weight.device.index))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x_sent, x_lang, x_word_chars, x_word_case, x_lang_word, x_sent_len, x_word_len, x_sent_masks, x_word_masks, y_upos, y_xpos, y_attrs = batch
        p_upos, p_xpos, p_attrs = self.forward(x_sent, x_lang, x_word_chars, x_word_case, x_lang_word, x_sent_len,
                                               x_word_len,
                                               x_sent_masks,
                                               x_word_masks)

        loss_upos = F.cross_entropy(p_upos.view(-1, p_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_xpos = F.cross_entropy(p_xpos.view(-1, p_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_attrs = F.cross_entropy(p_attrs.view(-1, p_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)
        return (loss_upos + loss_attrs + loss_xpos) / 3

    def validation_step(self, batch, batch_idx):
        x_sent, x_lang, x_word_chars, x_word_case, x_lang_word, x_sent_len, x_word_len, x_sent_masks, x_word_masks, y_upos, y_xpos, y_attrs = batch
        p_upos, p_xpos, p_attrs = self.forward(x_sent, x_lang, x_word_chars, x_word_case, x_lang_word, x_sent_len,
                                               x_word_len,
                                               x_sent_masks,
                                               x_word_masks)

        loss_upos = F.cross_entropy(p_upos.view(-1, p_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_xpos = F.cross_entropy(p_xpos.view(-1, p_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_attrs = F.cross_entropy(p_attrs.view(-1, p_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)
        loss = (loss_upos + loss_attrs + loss_xpos) / 3
        upos_ok = 0
        xpos_ok = 0
        attrs_ok = 0
        total = 0

        pred_upos = torch.argmax(p_upos).detach().cpu().numpy()
        pred_xpos = torch.argmax(p_xpos).detach().cpu().numpy()
        pred_attrs = torch.argmax(p_attrs).detach().cpu().numpy()
        tar_upos = y_upos.detach().cpu().numpy()
        tar_xpos = y_xpos.detach().cpu().numpy()
        tar_attrs = y_attrs.detach().cpu().numpy()
        sl = x_sent_len.detach().cpu().numpy()

        for iSent in range(x_sent.shape[0]):
            for iWord in range(sl[iSent]):
                total += 1
                if pred_upos[iSent, iWord] == tar_upos[iSent, iWord]:
                    upos_ok += 1
                if pred_xpos[iSent, iWord] == tar_xpos[iSent, iWord]:
                    xpos_ok += 1
                if pred_attrs[iSent, iWord] == tar_attrs[iSent, iWord]:
                    attrs_ok += 1

        return {'loss': loss, 'upos_ok': upos_ok, 'xpos_ok': xpos_ok, 'attrs_ok': attrs_ok, 'total': total}

    def validation_epoch_end(self, outputs):
        valid_loss_total = 0
        upos_ok = 0
        xpos_ok = 0
        attrs_ok = 0
        total = 0
        for out in outputs:
            valid_loss_total += out['loss']
            total += out['total']
            upos_ok += out['upos_ok']
            attrs_ok += out['attrs_ok']
            xpos_ok += out['xpos_ok']

        self.log('val/loss', valid_loss_total / len(outputs))
        self.log('val/upos', upos_ok / total)
        self.log('val/xpos', xpos_ok / total)
        self.log('val/attrs', attrs_ok / total)


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

    config = TaggerConfig()
    model = Tagger(config=config, encodings=enc)
    # training
    trainer = pl.Trainer(gpus=0, num_nodes=1, limit_train_batches=0.5, accelerator="ddp_cpu")
    trainer.fit(model, train_loader, val_loader)
