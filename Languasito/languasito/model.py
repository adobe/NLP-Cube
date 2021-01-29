import sys

sys.path.append('')
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import *
import random

from languasito.utils import Encodings
from languasito.modules import WordGram, LinearNorm, CosineLoss


class Languasito(pl.LightningModule):
    def __init__(self, encodings: Encodings):
        super().__init__()
        NUM_FILTERS = 512
        RNN_SIZE = 256

        self._wg = WordGram(len(encodings.char2int), num_langs=1, num_filters=512, num_layers=5)
        self._rnn_fw = nn.LSTM(NUM_FILTERS // 2, RNN_SIZE, num_layers=3, batch_first=True, bidirectional=False)
        self._rnn_bw = nn.LSTM(NUM_FILTERS // 2, RNN_SIZE, num_layers=3, batch_first=True, bidirectional=False)
        self._linear_out = LinearNorm(RNN_SIZE * 2, NUM_FILTERS // 2)
        self._early_stop_meta_val = 0
        self._res = {"b_loss": 9999}
        self._start_stop = nn.Embedding(2, NUM_FILTERS // 2)
        self._epoch_results = None
        self._ge2e_word = CosineLoss()
        self._ge2e_sent = CosineLoss()

    def forward(self, X):
        x_words_chars = X['x_word_char']
        x_words_case = X['x_word_case']
        x_lang_word = X['x_lang_word']
        x_sent_len = X['x_sent_len']
        x_word_len = X['x_word_len']
        x_word_masks = X['x_word_masks']
        x_max_len = X['x_max_len']
        char_emb_packed = self._wg(x_words_chars, x_words_case, x_lang_word, x_word_masks, x_word_len)

        blist_char = []
        sl = x_sent_len.cpu().numpy()
        pos = 0
        for ii in range(x_sent_len.shape[0]):
            slist_char = []
            slist_char.append(
                self._start_stop(torch.zeros((1), dtype=torch.long, device=self._get_device())))
            for jj in range(sl[ii]):
                slist_char.append(char_emb_packed[pos, :].unsqueeze(0))
                pos += 1

            slist_char.append(
                self._start_stop(torch.ones((1), dtype=torch.long, device=self._get_device())))

            for jj in range(x_max_len - sl[ii]):
                slist_char.append(torch.zeros((1, 512 // 2),
                                              device=self._get_device(), dtype=torch.float))

            sent_emb = torch.cat(slist_char, dim=0)
            blist_char.append(sent_emb.unsqueeze(0))

        char_emb = torch.cat(blist_char, dim=0)
        out_fw, _ = self._rnn_fw(char_emb)
        out_bw, _ = self._rnn_bw(torch.flip(char_emb, [1]))
        out_bw = torch.flip(out_bw, [1])
        lexical = char_emb[:, 1:-1, :]
        context = torch.cat([out_fw[:, :-2, :], out_bw[:, 2:, :]], dim=-1)
        context = torch.tanh(self._linear_out(context))
        # embeds = self._linear_out(context)
        # norm = embeds.norm(p=2, dim=-1, keepdim=True)
        # embeds_normalized = embeds.div(norm)
        # context = embeds_normalized
        concat = torch.cat([lexical, context], dim=-1)

        # fw_lst = []
        # bw_lst = []
        # for ii in range(x_sent_len.shape[0]):
        #     fw_lst.append(out_fw[ii, sl[ii] + 1].unsqueeze(0))
        #     bw_lst.append(out_bw[ii, 0].unsqueeze(0))
        # bw_lst = torch.cat(bw_lst, dim=0)
        # fw_lst = torch.cat(fw_lst, dim=0)
        # sent = torch.cat([bw_lst, fw_lst], dim=-1)

        y = {'lexical': lexical, 'context': context, 'emb': concat}  # , 'sent': sent}
        return y

    def training_step(self, batch, batch_idx):
        Y = self.forward(batch)
        y = Y['emb']
        y_lexical = Y['lexical']
        y_context = Y['context']
        # y_sent = Y['sent']

        sl = batch['x_sent_len'].detach().cpu().numpy()

        word_repr = []
        # sent_repr = y_sent
        for ii in range(sl.shape[0]):
            for jj in range(sl[ii]):
                if True:  # random.random() < 0.15:
                    word_repr.append(y_lexical[ii, jj].unsqueeze(0))
                    word_repr.append(y_context[ii, jj].unsqueeze(0))

        word_repr = torch.cat(word_repr, dim=0)
        word_repr = word_repr.reshape(-1, 2, word_repr.shape[1])
        loss_word = self._ge2e_word(word_repr)

        # sent_repr = sent_repr.reshape(-1, 2, sent_repr.shape[1])
        # loss_sent = self._ge2e_sent(sent_repr)
        return loss_word

    def validation_step(self, batch, batch_idx):
        Y = self.forward(batch)
        y = Y['emb']
        y_lexical = Y['lexical']
        y_context = Y['context']
        # y_sent = Y['sent']

        sl = batch['x_sent_len'].detach().cpu().numpy()

        word_repr = []
        # sent_repr = y_sent
        for ii in range(sl.shape[0]):
            for jj in range(sl[ii]):
                if True:  # random.random() < 0.15:
                    word_repr.append(y_lexical[ii, jj].unsqueeze(0))
                    word_repr.append(y_context[ii, jj].unsqueeze(0))

        word_repr = torch.cat(word_repr, dim=0)
        word_repr = word_repr.reshape(-1, 2, word_repr.shape[1])
        loss_word = self._ge2e_word(word_repr)

        # sent_repr = sent_repr.reshape(-1, 2, sent_repr.shape[1])
        # loss_sent = self._ge2e_sent(sent_repr)
        return {'total_loss': loss_word}

    def validation_epoch_end(self, outputs: List[Any]) -> None:

        loss = 0
        for output in outputs:
            loss += output['total_loss']
        loss /= len(outputs)

        res = {'val_loss': loss}
        self._epoch_results = self._compute_early_stop(res)
        self.log('val/early_meta', self._early_stop_meta_val)
        self.log('val/loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

    def _compute_early_stop(self, res):
        if res["val_loss"] < self._res['b_loss']:
            self._early_stop_meta_val += 1
            self._res['b_loss'] = res["val_loss"]
            res['best_loss'] = True
        return res

    def _get_device(self):
        if self._start_stop.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._start_stop.weight.device.type, str(self._start_stop.weight.device.index))

    def load(self, filename: str):
        self.load_state_dict(torch.load(filename, map_location='cpu'))
