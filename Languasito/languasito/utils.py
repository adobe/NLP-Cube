import re
from torch.utils.data import Dataset
import json, os
from tqdm.autonotebook import tqdm as tqdm
import torch
import numpy as np
import random
import pickle
from collections import defaultdict
from tokenizers import Tokenizer


class LanguasitoTokenizer:
    def __init__(self, no_space_language=False):
        self._no_space_language = no_space_language

    def __call__(self, text):
        if self._no_space_language:
            return [ch for ch in text]
        else:
            toks = []
            tok = ''
            for ch in text:
                if not ch.isalnum() or ch == ' ':
                    tok = tok.strip()
                    if len(tok) != 0:
                        toks.append(tok)
                        tok = ''
                    if ch != ' ':
                        toks.append(ch)
                else:
                    tok += ch
            if tok.strip() != '':
                toks.append(tok)

            return toks


class LanguasitoCollate:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    def _encode_words(self, words):
        try:
            encoded_batch = self._tokenizer.encode_batch(words)
            seq_lens = [len(x.ids) for x in encoded_batch]
            max_seq_len = max(seq_lens)
            x_encoded = np.zeros((len(words), max_seq_len), dtype=np.longlong)
            for ii in range(x_encoded.shape[0]):
                for jj in range(x_encoded.shape[1]):
                    if jj < len(encoded_batch[ii].ids):
                        x_encoded[ii, jj] = encoded_batch[ii].ids[jj]
            return x_encoded, np.array(seq_lens, dtype=np.longlong)
        except:
            return None, None

    def collate_fn(self, X):
        x = [item['source_word'] for item in X]
        x_encoded, x_seq_lens = self._encode_words(x)
        if x_encoded is None:
            return []  # obviously there is a bug in wordpiece
        if 'target_word' in X[0] and X[0]['target_word'] is not None:
            y = [item['target_word'] for item in X]
            y_encoded, y_seq_lens = self._encode_words(y)
        else:
            y_encoded = y_seq_lens = None
        return {
            'x_ids': torch.tensor(x_encoded),
            'x_seq_lens': torch.tensor(x_seq_lens),
            'y_ids': y_encoded,
            'y_seq_lens': torch.tensor(y_seq_lens)
        }


def mp_job(data):
    no_space_lang, lines = data
    print(f"\t\ttokenizing {len(lines)} ...")
    _st = LanguasitoTokenizer(no_space_language=no_space_lang)
    filtered_lines = []
    # new_lines = []
    for line in lines:
        toks = _st(line)
        if len(toks) > 5 and len(toks) < 50:
            valid = True
            for tok in toks:
                if len(tok) > 20:
                    valid = False
                    break
            if valid:
                filtered_lines.append(toks)
                # new_lines.append(line)

    return filtered_lines


class LanguasitoDataset(Dataset):
    def __init__(self, filename: str = None):
        self._examples = []
        self._intervals = []
        self._total_examples = 0
        self.word_freqs = defaultdict()
        if filename is not None:
            self.load_file(filename)

    def load_file(self, filename: str):
        lines = open(filename).readlines()
        for line in lines:
            parts = line.split('\t')
            self._examples.append((str(parts[0]), str(parts[1]), int(parts[2])))
            if len(self._intervals) == 0:
                self._intervals.append((0, int(parts[2])))
            else:
                self._intervals.append((self._intervals[-1][1], self._intervals[-1][1] + int(parts[2])))
            word = str(parts[0])
            count = int(parts[2])
            self.word_freqs[word] = self.word_freqs.get(word, 0) + count
            self._total_examples += count

    def __len__(self):
        return self._total_examples

    def __getitem__(self, item):
        # binary search
        start = 0
        end = len(self._examples) - 1
        pivot = (start + end) // 2
        answer = pivot
        while start <= end:
            pivot = (start + end) // 2
            if self._intervals[pivot][0] <= item:
                answer = pivot
                start = pivot + 1
            else:
                end = pivot - 1

        # while self._intervals[pivot][0] > item or self._intervals[pivot][1] < item:
        #     if start == end or pivot == end or pivot == start:
        #         break
        #     if self._intervals[pivot][0] > item:
        #         end = pivot
        #     else:
        #         start = pivot
        #
        #     pivot = (start + end) // 2
        return {
            'source_word': self._examples[answer][0],
            'target_word': self._examples[answer][1]
        }


class Encodings:
    def __init__(self):
        pass
