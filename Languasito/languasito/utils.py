import re
from torch.utils.data import Dataset
import json
from tqdm.autonotebook import tqdm as tqdm
import torch
import numpy as np


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
    # def __call__(self, text):
    #     if self._no_space_language:
    #         return [ch for ch in text]
    #     else:
    #         punctuation = '''"’'()[]{}<>:,‒–—―…!.«»-?‘’“”;/⁄␠·&@*\\•^¤¢$€£¥₩₪†‡°¡¿¬#№%‰‱¶′§~¨_|¦⁂☞∴‽※"„”'''
    #         new_text = ''
    #         for ch in text:
    #             if re.match(u'[\u4e00-\u9fff]', ch):
    #                 new_text += ' ' + ch + ' '
    #             elif ch in punctuation:
    #                 new_text += ' ' + ch + ' '
    #             else:
    #                 new_text += ch
    #
    #         tmp = new_text.replace('  ', ' ')
    #         while tmp != new_text:
    #             new_text = tmp
    #             tmp = new_text.replace('  ', ' ')
    #         new_text = new_text.strip()
    #         return new_text.split(' ')


class LanguasitoDataset(Dataset):
    def __init__(self, no_space_lang=False):
        self._examples = []
        self._st = LanguasitoTokenizer(no_space_language=no_space_lang)

    def load_file(self, filename: str):
        import multiprocessing
        # lines = open(filename, encoding='utf-8').readlines()
        lines = []
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                lines.append(line)
                # if len(lines) % 1000000 == 0:
                #    print(len(lines))
                if len(lines) > 10000000:
                    break
        print("Filtering...")
        """
        def mp_worker(lines):
            return lines

        def update(*a):
            pbar.update(1)


        pbar = tqdm(total=8, unit="chunks")
        pool = multiprocessing.Pool(8)

        data = []
        results = []
        chunk = len(lines)//8
        for i in range(8):
            data.append(lines[i*chunk:(i+1)*chunk])

        for datachunk in data:
            results.extend( pool.apply_async(mp_worker, args=(datachunk,), callback=update).get() )

        print(len(results))
        print(results[0])

        pool.close()
        pool.join()
        pbar.close()
        """

        filtered_lines, o_lines = self._filter(lines)
        n = len(filtered_lines)
        for ii in range(n):
            tokenized = filtered_lines[ii]
            if o_lines[ii].startswith("<doc id="):
                continue
            if ii < n - 1 and not o_lines[ii + 1].startswith("<doc id="):
                ni = len(self._examples) + 1
            else:
                ni = len(self._examples) - 1
            self._examples.append([tokenized, ni])

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        sent1 = self._examples[item][0]
        sent2 = self._examples[self._examples[item][1]][0]
        return {'sent1': sent1, 'sent2': sent2}

    def _filter(self, lines):
        filtered_lines = []
        new_lines = []
        for line in tqdm(lines):
            toks = self._st(line)
            if len(toks) > 5 and len(toks) < 50:
                valid = True
                for tok in toks:
                    if len(tok) > 20:
                        valid = False
                        break
                if valid:
                    filtered_lines.append(toks)
                    new_lines.append(line)
        return filtered_lines, new_lines


def load_dataset(filename: str) -> LanguasitoDataset:
    dataset = LanguasitoDataset()
    lines = open(filename).readlines()
    for ii in tqdm(range(len(lines)), desc='Loading dataset "{0}"'.format(filename), ncols=100):
        fname = lines[ii].strip()
        dataset.load_file(fname)
    return dataset


class Encodings:
    def __init__(self, max_vocab_size: int = 10000, min_word_occ: int = 5, min_char_occ: int = 20):
        self._max_vocab_size = max_vocab_size
        self._min_word_occ = min_word_occ
        self._min_char_occ = min_char_occ
        self.char2int = {'<PAD>': 0, '<UNK>': 1, '<SOT>': 2, '<EOT>': 3}
        self.word2int = {}

    def load(self, filename: str):
        json_obj = json.load(open(filename))
        self.char2int = json_obj['char2int']
        if 'word2int' in json_obj:
            self.word2int = json_obj['word2int']

    def save(self, filename: str, full: bool = True):
        json_obj = {'char2int': self.char2int}
        if full:
            json_obj['word2int'] = self.word2int
        json.dump(json_obj, open(filename, 'w'))

    def update(self, dataset: LanguasitoDataset):
        word2count = {}
        char2count = {}
        n = len(dataset)
        for ii in tqdm(range(n), ncols=100, desc='Updating encodings'):
            tokenized = dataset[ii]['sent1']
            for tok in tokenized:
                tok_lower = tok.lower()
                if tok_lower not in word2count:
                    word2count[tok_lower] = 1
                else:
                    word2count[tok_lower] += 1

                for ch in tok_lower:
                    if ch not in char2count:
                        char2count[ch] = 1
                    else:
                        char2count[ch] += 1

        # sort dict
        sorted_words = [k for k, v in sorted(word2count.items(), key=lambda item: item[1], reverse=True)]
        sorted_words = sorted_words[:min(len(sorted_words), self._max_vocab_size)]
        for w in sorted_words:
            if word2count[w] > self._min_word_occ:
                self.word2int[w] = len(self.word2int)

        for ch in char2count:
            if char2count[ch] > self._min_char_occ:
                self.char2int[ch] = len(self.char2int)


class LanguasitoCollate:
    def __init__(self, encodings: Encodings, live: bool = False):
        self._encodings = encodings
        self._live = live

    def collate_fn(self, batch):
        if not self._live:
            new_batch = []
            for b in batch:
                new_batch.append(b['sent1'])
                new_batch.append(b['sent2'])
            batch = new_batch
        a_sent_len = [len(sent) for sent in batch]
        a_word_len = []
        for sent in batch:
            for word in sent:
                a_word_len.append(len(word))
        x_sent_len = np.array(a_sent_len, dtype=np.long)
        x_word_len = np.array(a_word_len, dtype=np.long)
        max_sent_len = np.max(x_sent_len)
        max_word_len = np.max(x_word_len)
        x_sent_masks = np.zeros((len(batch), max_sent_len), dtype=np.float)
        x_word_masks = np.zeros((x_word_len.shape[0], max_word_len + 2), dtype=np.float)

        x_word_char = np.zeros((x_word_len.shape[0], max_word_len + 2), dtype=np.long)
        x_word_case = np.zeros((x_word_len.shape[0], max_word_len + 2), dtype=np.long)
        c_word = 0
        x_lang_sent = np.zeros((len(batch)), dtype=np.long)
        x_lang_word = []

        for iSent in range(len(batch)):
            sent = batch[iSent]
            x_lang_sent[iSent] = 1
            for iWord in range(len(sent)):
                x_word_char[iWord, 0] = 2  # start of token
                word = sent[iWord]
                x_sent_masks[iSent, iWord] = 1
                x_lang_word.append(1)

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
                        x_word_char[c_word, iChar + 1] = self._encodings.char2int[ch]
                    else:
                        x_word_char[c_word, iChar + 1] = self._encodings.char2int['<UNK>']
                x_word_char[c_word, len(word) + 1] = 3  # end of token
                x_word_masks[c_word, len(word) + 1] = 1
                c_word += 1

        x_lang_word = np.array(x_lang_word)
        response = {
            'x_word_char': torch.tensor(x_word_char),
            'x_word_case': torch.tensor(x_word_case),
            'x_lang_word': torch.tensor(x_lang_word),
            'x_sent_len': torch.tensor(x_sent_len),
            'x_word_len': torch.tensor(x_word_len),
            'x_sent_masks': torch.tensor(x_sent_masks),
            'x_word_masks': torch.tensor(x_word_masks),
            'x_max_len': max_sent_len
        }

        return response
