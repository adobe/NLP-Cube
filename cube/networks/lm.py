import sys
import tqdm
from abc import abstractmethod

import torch
import numpy as np
from typing import *

sys.path.append('')
from transformers import AutoTokenizer
from transformers import AutoModel
from cube.io_utils.objects import Sentence, Document
import fasttext
import fasttext.util


class LMHelper:
    def __init__(self):
        pass

    @abstractmethod
    def get_embedding_size(self):
        pass

    @abstractmethod
    def apply(self, document: Document):
        pass

    @abstractmethod
    def apply_raw(self, batch):
        pass


class LMHelperFT(LMHelper):
    def __init__(self, device: str = 'cpu', model: str = None):
        from pathlib import Path
        home = str(Path.home())
        filename = '{0}/.fasttext/cc.{1}.300.bin'.format(home, model)
        import os
        if not os.path.exists(filename):
            fasttext.util.download_model(model, if_exists='ignore')  # English
            in_file = "cc.{0}.300.bin".format(model)
            import shutil
            import pathlib
            print("Creating " + "{0}/.fasttext/".format(home))
            pathlib.Path("{0}/.fasttext/".format(home)).mkdir(parents=True, exist_ok=True)
            shutil.move(in_file, filename)
        self._fasttext = fasttext.load_model(filename)

    def get_embedding_size(self):
        return [300]

    def apply(self, document: Document):
        for ii in tqdm.tqdm(range(len(document.sentences)), desc="Pre-computing embeddings", unit="sent"):
            for jj in range(len(document.sentences[ii].words)):
                document.sentences[ii].words[jj].emb = [self._fasttext.get_word_vector(
                    document.sentences[ii].words[jj].word)]

    def apply_raw(self, batch):
        embeddings = []
        for ii in range(len(batch)):
            c_emb = []
            for jj in range(len(batch[ii])):
                c_emb.append(self._fasttext.get_word_vector(batch[ii][jj]))
            embeddings.append(c_emb)
        return embeddings


class LMHelperLanguasito(LMHelper):
    def __init__(self, device: str = 'cpu', model: str = None):
        if model is None:
            print("UserWarning: No languasito model was specified. Instance will fail")
        from languasito.api import LanguasitoAPI
        self._languasito = LanguasitoAPI.load(model)
        self._languasito.to(device)

    def get_embedding_size(self):
        # TODO: a better way to get the embedding size (right now it is hardcoded)
        return [1024]

    def apply(self, document: Document):
        BATCH_SIZE = 8
        num_batches = len(document.sentences) // BATCH_SIZE
        if len(document.sentences) % BATCH_SIZE != 0:
            num_batches += 1

        for iBatch in tqdm.tqdm(range(num_batches), desc="Pre-computing embeddings", unit="sent"):
            start = iBatch * BATCH_SIZE
            stop = min(iBatch * BATCH_SIZE + BATCH_SIZE, len(document.sentences))
            batch = []
            for ii in range(start, stop):
                cb = []
                for w in document.sentences[ii].words:
                    cb.append(w.word)
                batch.append(cb)
            embeddings = self._languasito(batch)
            for ii in range(len(batch)):
                for jj in range(len(batch[ii])):
                    document.sentences[ii + start].words[jj].emb = [embeddings[ii][jj]]

    def apply_raw(self, batch):
        BATCH_SIZE = 8
        num_batches = len(batch) // BATCH_SIZE
        if len(batch) % BATCH_SIZE != 0:
            num_batches += 1

        for iBatch in range(num_batches):
            start = iBatch * BATCH_SIZE
            stop = min(iBatch * BATCH_SIZE + BATCH_SIZE, len(batch))
            tb = []
            for ii in range(start, stop):
                cb = []
                for w in batch[ii]:
                    cb.append(w)
                tb.append(cb)
            embeddings = self._languasito(batch)

        return embeddings


class LMHelperHF(LMHelper):
    def __init__(self, device: str = 'cpu', model: str = None):
        if model is None:
            self._splitter = AutoTokenizer.from_pretrained('xlm-roberta-base')
            self._xlmr = AutoModel.from_pretrained('xlm-roberta-base',
                                                   output_hidden_states=True)
        else:
            self._splitter = AutoTokenizer.from_pretrained(model)
            self._xlmr = AutoModel.from_pretrained(model, output_hidden_states=True)
        self._xlmr.eval()
        self._xlmr.to(device)
        self._device = device
        tmp = self._xlmr(torch.tensor([[100]], device=device))
        h_state_size = tmp['hidden_states'][0].shape[-1]
        self._emb_size = [h_state_size for _ in range(len(tmp['hidden_states']))]

    def get_embedding_size(self):
        # TODO: a better way to get the embedding size (right now it is hardcoded)
        return self._emb_size

    def _compute_we(self, batch: [Sentence]):
        # XML-Roberta

        # convert all words into wordpiece indices
        word2pieces = {}
        new_sents = []
        START = 0
        PAD = 1
        END = 2
        for ii in range(len(batch)):
            c_sent = [START]
            pos = 1
            for jj in range(len(batch[ii].words)):
                word = batch[ii].words[jj].word
                pieces = self._splitter(word)['input_ids'][1:-1]
                word2pieces[(ii, jj)] = []
                for piece in pieces:
                    c_sent.append(piece)
                    word2pieces[(ii, jj)].append([ii, pos])
                    pos += 1
            c_sent.append(END)
            new_sents.append(c_sent)
        max_len = max([len(s) for s in new_sents])
        input_ids = np.ones((len(new_sents), max_len), dtype='long') * PAD  # pad everything
        for ii in range(input_ids.shape[0]):
            for jj in range(input_ids.shape[1]):
                if jj < len(new_sents[ii]):
                    input_ids[ii, jj] = new_sents[ii][jj]
        with torch.no_grad():
            x = torch.tensor(input_ids, device=self._device)
            max_s_len = x.shape[1]
            count = max_s_len // 512

            if max_s_len % 512 != 0:
                count += 1
            we_list = []
            for index in range(count):
                out = self._xlmr(x[:, index * 512:min(x.shape[1], index * 512 + 512)], return_dict=True)
                we = torch.cat(out['hidden_states'], dim=-1).detach().cpu()
                we_list.append(we)
            we = torch.cat(we_list, dim=1).numpy()

        word_emb = []
        for ii in range(len(batch)):
            for jj in range(len(batch[ii].words)):
                pieces = word2pieces[ii, jj]
                if len(pieces) != 0:
                    m = we[pieces[0][0], pieces[0][1]]
                    for zz in range(len(pieces) - 1):
                        m += we[pieces[zz][0], pieces[zz][1]]
                    m = m / len(pieces)
                else:
                    m = np.zeros((768 * 13), dtype='float')
                word_emb.append(m)
        # word_emb = torch.cat(word_emb, dim=0)

        return word_emb

    def apply(self, doc: Document):
        import tqdm
        for sent in doc.sentences:  # tqdm.tqdm(doc.sentences, desc="Pre-computing embeddings", unit="sent"):
            wemb = self._compute_we([sent])
            for ii in range(len(wemb)):
                ww = wemb[ii]
                www = []
                for kk in range(13):
                    www.append(ww[kk * 768:kk * 768 + 768])
                sent.words[ii].emb = www

    def apply_raw(self, batch):
        pass


class LMHelperDummy(LMHelper):
    def __init__(self, device: str = 'cpu', model: str = None):
        pass

    def get_embedding_size(self):
        return [1]

    def apply(self, document: Document):
        for ii in tqdm.tqdm(range(len(document.sentences)), desc="Pre-computing embeddings", unit="sent"):
            for jj in range(len(document.sentences[ii].words)):
                document.sentences[ii].words[jj].emb = [[1.0]]

    def apply_raw(self, batch):
        embeddings = []
        for ii in range(len(batch)):
            c_emb = []
            for jj in range(len(batch[ii])):
                c_emb.append([1.0])
            embeddings.append(c_emb)
        return embeddings

