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
        self._outputs = []
        self._res = {'b_loss': 9999}
        self._early_stop_meta_val = 0
        self._vocab_size = len(tokenizer.get_vocab())

        self._we = nn.Embedding(len(tokenizer.get_vocab()), 512, padding_idx=0)

        # self._wg = WordGram(len(tokenizer.get_vocab()), num_langs=1, num_filters=512, num_layers=4)
        # self._decoder = nn.Linear(256, 500)

    def forward(self, X):
        # hidden = self._wg(X['x_ids'], X['x_seq_lens'], X['x_masks'])
        emb = self._we(X['x_ids'])
        hidden = emb.sum(dim=1) / X['x_seq_lens'].unsqueeze(1)

        # hidden = self._decoder(hidden)
        hidden = torch.nn.functional.normalize(hidden)

        return hidden

    def training_step(self, batch, batch_idx):
        if len(batch) == 0:
            return 0
        hidden = self.forward(batch)
        src_embeddings = hidden[batch['source_index']]
        dst_embeddings = hidden[batch['destination_index']]
        targets = batch['targets']

        # normalize
        loss = torch.nn.functional.cosine_embedding_loss(src_embeddings, dst_embeddings, targets)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 0:
            return 0
        hidden = self.forward(batch)
        src_embeddings = hidden[batch['source_index']]
        dst_embeddings = hidden[batch['destination_index']]
        targets = batch['targets']
        # normalize
        loss = torch.nn.functional.cosine_embedding_loss(src_embeddings, dst_embeddings, targets).item()
        self._outputs.append({'total_loss': loss})
        return loss

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
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def _compute_early_stop(self, res):
        if res["val_loss"] < self._res['b_loss']:
            if self._res['b_loss'] == 9999:
                self._res['b_loss'] = 9998
            else:
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
    from languasito.utils import LanguasitoCollate, LanguasitoDataset, LanguasitoWordGramTokenizer

    wp = LanguasitoWordGramTokenizer('en_wiki.wordpiece')
    collate = LanguasitoCollate(wp)

    model = Languasito(wp)
    model.load('en_wiki.last')
    model.eval()
    model.to('mps')

    # build lexion
    dataset = LanguasitoDataset('../../docubert/en_wiki.train')
    word2vec = {}
    from tqdm import tqdm

    index = 0
    BS = 512
    batches = len(dataset.word_freqs) // BS
    wl = [w for w in dataset.word_freqs]
    if len(dataset.word_freqs) % BS != 0:
        batches += 1

    for ii in tqdm(range(batches)):
        start = ii * BS
        stop = min(ii * BS + BS, len(wl))
        mini_batch = wl[start:stop]
        X = collate.collate_fn([{'source_word': word, 'positive_words': [], 'negative_words': []} for word
                                in mini_batch])
        for key in X:
            if isinstance(X[key], torch.Tensor):
                X[key] = X[key].to('mps')
        with torch.no_grad():
            vector = model(X).detach().cpu().numpy()
            for jj in range(vector.shape[0]):
                word2vec[mini_batch[jj]] = vector[jj]

    while True:
        word = input("Word: ")
        if word == "/quit":
            break

        X = collate.collate_fn([{'source_word': word, 'positive_words': [], 'negative_words': []}])
        for key in X:
            if isinstance(X[key], torch.Tensor):
                X[key] = X[key].to('mps')
        with torch.no_grad():
            vector = model(X).detach().cpu().numpy()[0]
        tk = _get_top_k(vector, word2vec, 20)
        for t in tk:
            print(f"\t{t}")
