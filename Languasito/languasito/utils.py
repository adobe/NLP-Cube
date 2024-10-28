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
        encoded_batch = self._tokenizer.encode_batch(words)
        seq_lens = [len(x.ids) for x in encoded_batch]
        max_seq_len = max(seq_lens)
        x_encoded = np.zeros((len(words), max_seq_len), dtype=np.longlong)
        for ii in range(x_encoded.shape[0]):
            for jj in range(x_encoded.shape[1]):
                if jj < len(encoded_batch[ii].ids):
                    x_encoded[ii, jj] = encoded_batch[ii].ids[jj]
        return x_encoded, np.array(seq_lens, dtype=np.longlong)

    def _encode_targets(self, y_targets):
        y_encoded = np.zeros((len(y_targets), len(self._tokenizer.get_vocab())), dtype=np.longlong)
        y_seq_lens = np.zeros((len(y_targets)), dtype=np.longlong)
        index = 0

        for y_target in y_targets:
            counts = [item[1] for item in y_target]
            list_of_ids = self._tokenizer.encode_batch([item[0] for item in y_target])
            ii = 0
            for ids in list_of_ids:
                ids = ids.ids
                y_seq_lens[index] += len(ids)
                for id in ids:
                    y_encoded[index, id] += counts[ii]
                ii += 1
            index += 1
        return y_encoded, y_seq_lens

    def collate_fn(self, X):
        x = [item['source_word'] for item in X]
        x_encoded, x_seq_lens = self._encode_words(x)
        y_encoded = [0]
        y_seq_lens = [0]
        if 'target_words' in X[0]:
            y_targets = [item['target_words'] for item in X]
            y_encoded, y_seq_lens = self._encode_targets(y_targets)

        return {
            'x_ids': torch.tensor(x_encoded),
            'x_seq_lens': torch.tensor(x_seq_lens),
            'y_targets': torch.tensor(y_encoded, dtype=torch.float),
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
        self._word2word = {}
        self._int2word = {}
        self._total_examples = 0
        self.word_freqs = defaultdict()
        if filename is not None:
            self.load_file(filename)

    def load_file(self, filename: str):
        lines = open(filename).readlines()
        for line in lines:
            parts = line.split('\t')
            src_word = parts[1]
            dst_word = parts[0]
            count = int(parts[2])
            if src_word not in self._word2word:
                self._word2word[src_word] = []
            self._word2word[src_word].append((dst_word, count))
            word = str(parts[0])
            self.word_freqs[word] = self.word_freqs.get(word, 0) + count
            self._total_examples += count
        for word in self._word2word:
            self._int2word[len(self._int2word)] = word

    def __len__(self):
        return len(self._int2word)

    def __getitem__(self, item):
        word = self._int2word[item]
        return {
            'source_word': word,
            'target_words': self._word2word[word]
        }


class LanguasitoDatasetClassical(Dataset):
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
        return {
            'source_word': self._examples[answer][0],
            'target_word': self._examples[answer][1]
        }


class Encodings:
    def __init__(self):
        pass
