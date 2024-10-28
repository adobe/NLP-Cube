import sys

sys.path.append('')
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import *
import numpy as np
import random

from tokenizers import Tokenizer
from languasito.modules import WordGram, LinearNorm, CosineLoss, WordDecoder


class Languasito(pl.LightningModule):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__()
        NUM_FILTERS = 512
        RNN_SIZE = 256
        CHAR_EMB_SIZE = 128
        ATT_DIM = 64
        NUM_HEADS = 8
        self._outputs = []
        self._res = {'b_loss': 9999}
        self._early_stop_meta_val = 0
        self._vocab_size = len(tokenizer.get_vocab())

        self._wg = WordGram(len(tokenizer.get_vocab()), num_langs=1, num_filters=512, num_layers=5)
        self._decoder = nn.Linear(256, len(tokenizer.get_vocab()))

    def forward(self, X, return_proj=False):
        hidden = self._wg(X['x_ids'], X['x_seq_lens'])
        hidden = torch.nn.functional.normalize(hidden)
        if return_proj:
            proj = torch.softmax(self._decoder(hidden), dim=-1)
            return hidden, proj
        else:
            return hidden

    # def _get_targets(self, target_ids):
    #     target = torch.zeros((len(target_ids), self._vocab_size), dtype=torch.float, device=self._get_device())
    #     indices = []
    #     for ii in range(len(target_ids)):
    #         for t_id in target_ids[ii]:
    #             indices.append([ii, t_id])
    #
    #     indices = torch.tensor(indices, device=self._get_device()).transpose(0, 1)
    #     target[indices[0], indices[1]] = 1
    #     return target

    def training_step(self, batch, batch_idx):
        if len(batch) == 0:
            return 0
        hidden, proj = self.forward(batch, return_proj=True)
        targets = batch['y_targets']
        # normalize
        targets = torch.nn.functional.normalize(targets)
        loss = torch.nn.functional.kl_div(proj, targets, reduction='sum', log_target=False)
        loss = loss / batch['y_seq_lens'].sum()
        self.log("loss", -loss, prog_bar=True)
        return -loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 0:
            return 0
        hidden, proj = self.forward(batch, return_proj=True)
        targets = batch['y_targets']
        # normalize
        targets = torch.nn.functional.normalize(targets)
        loss = torch.nn.functional.kl_div(proj, targets, reduction='sum', log_target=False)
        loss = loss / batch['y_seq_lens'].sum()
        self._outputs.append({'total_loss': -loss.item()})
        return -loss

    def on_validation_epoch_end(self) -> None:
        outputs = self._outputs
        self._outputs = []
        loss = 0
        for output in outputs:
            loss += output['total_loss']
        loss /= len(outputs) + 1

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
        if self._decoder.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._decoder.weight.device.type, str(self._decoder.weight.device.index))

    def load(self, filename: str):
        self.load_state_dict(torch.load(filename, map_location='cpu'))

    def save(self, filename: str):
        torch.save(self.state_dict(), filename)


def _get_top_k(vector, word2vec, top_k=10):
    distances = {}
    for word in word2vec:
        tv = word2vec[word]
        # dist = ((tv - vector) ** 2).mean()
        dist = np.dot(tv, vector) / (np.linalg.norm(tv) * np.linalg.norm(vector))

        distances[word] = dist

    sorted_vals = [(k, v) for k, v in sorted(distances.items(), key=lambda item: item[1], reverse=True)]
    return sorted_vals[:top_k]


if __name__ == '__main__':
    from tokenizers import Tokenizer
    from languasito.utils import LanguasitoCollate, LanguasitoDataset

    wp = Tokenizer.from_file('ro_wiki.wordpiece')
    collate = LanguasitoCollate(wp)

    model = Languasito(wp)
    model.load('ro_wiki.last')
    model.eval()

    # build lexion
    dataset = LanguasitoDataset('../../docubert/ro_wiki.train')
    word2vec = {}
    from tqdm import tqdm

    index = 0
    for word in tqdm(dataset.word_freqs):
        index += 1
        X = collate.collate_fn([{'source_word': word, 'target_word': word}])
        with torch.no_grad():
            vector = model(X).detach().cpu().numpy()[0]
            word2vec[word] = vector

        if index == 0:
            break

    while True:
        word = input("Word: ")
        if word == "/quit":
            break

        X = collate.collate_fn([{'source_word': word, 'target_word': word}])
        with torch.no_grad():
            vector = model(X).detach().cpu().numpy()[0]
        tk = _get_top_k(vector, word2vec, 10)
        for t in tk:
            print(f"\t{t}")
