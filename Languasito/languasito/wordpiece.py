import sys

from sortedcontainers import SortedDict
from collections import defaultdict
import json


class WordPiece:
    def __init__(self, filename: str = None):
        if filename is not None:
            self.load(filename)
        self.alphabet = None
        self.vocab = None

    def load(self, filename: str):
        data = json.load(open(filename))
        self.alphabet = data['alphabet']
        self.vocab = data['vocab']

    def save(self, filename: str):
        obj = {
            'alphabet': self.alphabet,
            'vocab': self.vocab
        }
        json.dump(obj, open(filename, 'w'), indent=4)

    def _compute_pair_scores(self, splits, word_freqs):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    def _merge_pair(self, a, b, splits, word_freqs):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits

    def compute(self, word_freqs, vocab_size=10000):
        alphabet = []
        for word in word_freqs.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")

        alphabet.sort()
        self.alphabet = alphabet

        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in word_freqs.keys()
        }

        index = 0
        sys.stdout.write(f'\t::vocab size is {len(vocab)}\n')
        sys.stdout.flush()
        while len(vocab) < vocab_size:
            index += 1
            if index % 100 == 0:
                sys.stdout.write(f'\t::vocab size is {len(vocab)}\n')
                sys.stdout.flush()
            scores = self._compute_pair_scores(splits, word_freqs)
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            splits = self._merge_pair(*best_pair, splits, word_freqs)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            vocab.append(new_token)
        self.vocab = vocab

    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens
