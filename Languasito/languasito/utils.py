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


class TokenizerResult:
    ids = []


class LanguasitoWordGramTokenizer:
    def __init__(self, filename: str = None):
        self._tok2int = {'PADDING_INDEX': 0, 'UNKOWN_INDEX': 1}
        self._tok_list = ['PADDING_INDEX', 'UNKOWN_INDEX']
        if filename is not None:
            self.load(filename)

    def _get_all_ngrams(self, word, min_size=3, max_size=6, use_dict=False):
        word = f'<{word}>'
        ngrams = []
        for ii in range(len(word) - min_size + 1):
            for n_range in range(min_size, max_size):
                if n_range <= len(word) - ii:
                    ngram = word[ii:ii + n_range]
                    if not use_dict:
                        ngrams.append(ngram)
                    elif ngram in self._tok2int:
                        ngrams.append(ngram)
        if len(ngrams) == 0:
            ngrams.append('UNKOWN_INDEX')
        return ngrams

    def train_from_iterator(self, iterator, length=0, threshold=2):
        ngram2count = {}

        for ii in tqdm(range(length)):
            word = next(iterator)
            ngrams = self._get_all_ngrams(word)
            for ngram in ngrams:
                ngram2count[ngram] = ngram2count.get(ngram, 0) + 1
        for ngram in ngram2count:
            if ngram2count[ngram] >= threshold:
                self._tok2int[ngram] = len(self._tok2int)
                self._tok_list.append(ngram)

    def save(self, filename: str):
        json.dump({'tok2int': self._tok2int}, open(filename, 'w'), indent=4)

    def load(self, filename: str):
        obj = json.load(open(filename))
        self._tok2int = obj['tok2int']
        self._tok_list = ['' for _ in range(len(self._tok2int))]
        for tok in self._tok2int:
            self._tok_list[self._tok2int[tok]] = tok

    def encode_batch(self, words):
        res = []
        for word in words:
            ngrams = self._get_all_ngrams(word, use_dict=True)
            ids = [self._tok2int[tok] for tok in ngrams]
            tr = TokenizerResult()
            tr.ids = ids
            res.append(tr)

        return res

    def get_vocab(self):
        return self._tok2int


class LanguasitoCollate:
    def __init__(self, tokenizer: LanguasitoWordGramTokenizer):
        self._tokenizer = tokenizer

    def _encode_words(self, words):
        encoded_batch = self._tokenizer.encode_batch(words)
        seq_lens = [len(x.ids) for x in encoded_batch]
        max_seq_len = max(seq_lens)
        x_encoded = np.zeros((len(words), max_seq_len), dtype=np.longlong)
        x_masks = np.zeros((len(words), max_seq_len), dtype=np.float64)
        for ii in range(x_encoded.shape[0]):
            for jj in range(x_encoded.shape[1]):
                if jj < len(encoded_batch[ii].ids):
                    x_encoded[ii, jj] = encoded_batch[ii].ids[jj]
                    x_masks[ii, jj] = 1
        return x_encoded, np.array(seq_lens, dtype=np.longlong), x_masks

    def collate_fn(self, X):
        x = [item['source_word'] for item in X]

        source_word_list = [item['source_word'] for item in X]
        positive_word_list = []
        negative_word_list = []
        for item in X:
            for w in item['positive_words']:
                positive_word_list.append(w)
            for w in item['negative_words']:
                negative_word_list.append(w)
        x = source_word_list + positive_word_list + negative_word_list
        x_encoded, x_seq_lens, x_masks = self._encode_words(x)

        pos_samples = len(X[0]['positive_words'])
        neg_samples = len(X[0]['negative_words'])
        src_w = np.zeros(
            len(source_word_list) + len(source_word_list) * pos_samples + len(source_word_list) * neg_samples,
            dtype=np.longlong)
        dst_w = np.zeros(
            len(source_word_list) + len(source_word_list) * pos_samples + len(source_word_list) * neg_samples,
            dtype=np.longlong)
        target = np.zeros(
            len(source_word_list) + len(source_word_list) * pos_samples + len(source_word_list) * neg_samples,
            dtype=np.longlong)
        index = 0
        for ii in range(len(source_word_list)):
            for jj in range(pos_samples):
                src_w[index] = ii
                dst_w[index] = jj + len(source_word_list) + ii * pos_samples
                target[index] = 1
                index += 1

        for ii in range(len(source_word_list)):
            for jj in range(neg_samples):
                src_w[index] = ii
                dst_w[index] = jj + len(source_word_list) + len(source_word_list) * pos_samples + ii * neg_samples
                target[index] = -1
                index += 1

        return {
            'x_ids': torch.tensor(x_encoded, dtype=torch.long),
            'x_seq_lens': torch.tensor(x_seq_lens, dtype=torch.long),
            'x_masks': torch.tensor(x_masks, dtype=torch.float),
            'source_index': torch.tensor(src_w, dtype=torch.long),
            'destination_index': torch.tensor(dst_w, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
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
    def __init__(self, filename: str = None, negative_samples=5, positive_samples=4):
        self._word2word = {}
        self._int2word = {}
        self._total_examples = 0
        self._negative_samples = negative_samples
        self._positive_samples = positive_samples
        self.word_freqs = defaultdict()
        self._word_list = []
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
                self._word2word[src_word] = {'word_list': [], 'pos': 0}
            self._word2word[src_word]['word_list'].append((dst_word, count))
            word = str(parts[0])

            self.word_freqs[word] = self.word_freqs.get(word, 0) + count
            self._total_examples += count

        for word in self._word2word:
            self._int2word[len(self._int2word)] = word
            self._word_list.append(word)

    def __len__(self):
        return len(self._int2word)

    def __getitem__(self, item):
        word = self._int2word[item]
        # sample 5 positive words
        positive_words = []
        w2w = self._word2word[word]
        words = w2w['word_list']

        probs = np.array([w[1] for w in words], dtype=np.float64)
        probs = probs / probs.sum()
        words = [w[0] for w in words]
        all_pos = words
        for _ in range(self._positive_samples):
            positive_words.append(words[w2w['pos']])
            new_pos = w2w['pos'] + 1
            w2w['pos'] = new_pos % len(words)

            # positive_words.append(words[random.randint(0, len(words) - 1)])

        negative_words = []
        for _ in range(self._negative_samples):
            # nw = np.random.choice(self._word_list)
            while True:
                nw = self._word_list[random.randint(0, len(self._word_list) - 1)]
                if nw not in all_pos:
                    negative_words.append(nw)
                    break

        return {
            'source_word': word,
            'positive_words': positive_words,
            'negative_words': negative_words
        }


class Encodings:
    def __init__(self):
        pass
