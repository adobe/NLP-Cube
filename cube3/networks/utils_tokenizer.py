import sys
import torch
import numpy as np

sys.path.append('')
from typing import *
from abc import abstractmethod
from transformers import AutoModel, AutoTokenizer
from cube3.io_utils.encodings import Encodings
from cube3.io_utils.objects import Sentence
from cube3.networks.lm import LMHelperLanguasito, LMHelperFT
from languasito.utils import LanguasitoTokenizer


class TokenCollateTrain:
    def __init__(self):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    @abstractmethod
    def get_embeddings_size(self) -> int:
        pass


class TokenCollateTrainFTLanguasito(TokenCollateTrain):
    def __init__(self, encodings: Encodings, lm_model: str = None, lm_device: str = 'cuda:0', no_space_lang=False):
        self._encodings = encodings
        self._tokenizer = LanguasitoTokenizer(no_space_language=no_space_lang)
        self._emb_size = 0
        self._lm_model = lm_model
        self._lm_device = lm_device
        parts = lm_model.split(':')
        if parts[0] == 'fasttext':
            self._lm_helper = LMHelperFT(device=lm_device, model=parts[1])
            self._emb_size = 300
        elif parts[1] == 'languasito':
            self._lm_helper = LMHelperLanguasito(device=lm_device, model=parts[1])
            self._emb_size = 512
        else:
            print("UserWarning: unsupported LM type for tokenizer")

    def collate_fn(self, batch):
        START = 0
        END = 2
        PAD = 1
        max_x = 0
        x_input = []
        x_lang = []
        y_output = []
        y_offset = []
        y_len = []
        x_text = []
        x_lang_word = []
        x_sent_len = []

        x_word_embeddings = []
        a_word_len = []
        for example in batch:
            for qq in ['prev', 'main', 'next']:
                sent = example[qq]
                # toks, ids = self._tokenize(sent.text)
                toks = self._tokenizer(sent.text)
                for word in toks:
                    a_word_len.append(len(word))
                    x_lang_word.append(sent.lang_id)

        x_word_len = np.array(a_word_len, dtype=np.long)
        max_word_len = np.max(x_word_len)
        x_word_masks = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.float)
        x_word = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.long)
        x_word_case = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.long)
        c_word = 0
        for example in batch:
            sz = 0
            for qq in ['prev', 'main', 'next']:
                sent = example[qq]
                # toks, ids = self._tokenize(sent.text)
                toks = self._tokenizer(sent.text)
                lst = toks
                sz += len(lst)
                for word in lst:
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
            x_sent_len.append(sz)

        for example in batch:
            current_sentence = example['main']
            prev_sentence = example['prev']
            next_sentence = example['next']
            x_lang.append(current_sentence.lang_id + 1)
            # toks, ids = self._tokenize(prev_sentence.text)
            toks = self._tokenizer(prev_sentence.text)
            x_prev = toks
            # toks, ids = self._tokenize(next_sentence.text)
            toks = self._tokenizer(next_sentence.text)
            x_next = toks
            y_offset.append(len(x_prev) + 1)
            # c_toks, ids = self._tokenize(current_sentence.text)
            c_toks = self._tokenizer(current_sentence.text)
            x_main = c_toks
            y_len.append(len(x_main))
            x_len = len(x_prev) + len(x_main) + len(x_next)
            x_input.append([x_prev, x_main, x_next])
            x_text.append(c_toks)
            y_output.append(self._get_targets(current_sentence))
            if x_len > max_x:
                max_x = x_len

        # x_out = np.ones((len(batch), max_x + 2), dtype=np.long) * PAD
        # for ii in range(len(batch)):
        #     x_out[ii, 0] = START
        #     pos = 1
        #     x = x_input[ii][0]
        #     for jj in range(len(x)):
        #         x_out[ii, pos] = x[jj]
        #         pos += 1
        #     x = x_input[ii][1]
        #     for jj in range(len(x)):
        #         x_out[ii, pos] = x[jj]
        #         pos += 1
        #     x = x_input[ii][2]
        #     for jj in range(len(x)):
        #         x_out[ii, pos] = x[jj]
        #         pos += 1
        #     x_out[ii, pos] = END

        x_for_emb = []
        for example in x_input:
            toks = example[0] + example[1] + example[2]
            x_for_emb.append(toks)

        x_emb = self._lm_helper.apply_raw(x_for_emb)
        max_len = max([len(x) for x in x_emb])
        x_out = np.zeros((len(x_emb), max_len, self._emb_size), dtype=np.double)
        for ii in range(x_out.shape[0]):
            for jj in range(x_out.shape[1]):
                if jj < len(x_emb[ii]):
                    x_out[ii, jj, :] = x_emb[ii][jj]

        y_out = np.zeros((x_out.shape[0], x_out.shape[1]), dtype=np.long)
        for ii in range(x_out.shape[0]):
            for jj in range(y_len[ii]):
                index = y_offset[ii] + jj
                y_out[ii, index] = y_output[ii][jj]
        x_out = torch.tensor(x_out)
        x_lang = torch.tensor(x_lang)
        y_out = torch.tensor(y_out)
        y_offset = torch.tensor(y_offset)
        y_len = torch.tensor(y_len)
        x_word = torch.tensor(x_word)
        x_word_case = torch.tensor(x_word_case)
        x_word_masks = torch.tensor(x_word_masks)
        x_word_len = torch.tensor(x_word_len)
        x_lang_word = torch.tensor(x_lang_word)
        x_sent_len = torch.tensor(x_sent_len)

        return {'x_input': x_out, 'x_word_char': x_word, 'x_word_case': x_word_case, 'x_word_masks': x_word_masks,
                'x_word_len': x_word_len, 'x_word_lang': x_lang_word, 'x_text': x_text, 'x_lang': x_lang,
                'y_output': y_out, 'y_offset': y_offset, 'y_len': y_len, 'x_sent_len': x_sent_len}

    def _get_targets(self, sentence: Sentence):
        text = sentence.text
        toks = self._tokenizer(text)
        targets = [0 for _ in range(len(toks))]
        iToken = 0
        cl = 0
        for ii in range(len(targets)):
            target = 1  # nothing
            cl += len(toks[ii])
            if cl == len(sentence.tokens[iToken].text):
                iToken += 1
                cl = 0
                target = 2  # token
                if len(sentence.tokens[iToken].words) > 1:
                    target = 3  # multiword token
            if iToken == len(sentence.tokens):
                target = 4  # sentence end (+token)
            targets[ii] = target
        return targets

    def get_embeddings_size(self) -> int:
        return self._emb_size

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        if "_lm_helper" in state:
            del state["_lm_helper"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        parts = self._lm_model.split(':')
        if parts[0] == 'fasttext':
            self._lm_helper = LMHelperFT(device=self._lm_device, model=parts[1])
            self._emb_size = 300
        elif parts[1] == 'languasito':
            self._lm_helper = LMHelperLanguasito(device=self._lm_device, model=parts[1])
            self._emb_size = 512


class TokenCollateTrainHF(TokenCollateTrain):
    def __init__(self, encodings: Encodings, lm_model=None, lm_device='cuda:0', no_space_lang=False):
        if lm_model is None:
            lm_model = 'xlm-roberta-base'
        self._encodings = encodings  # this is currently not used - we keep it for future development
        self._tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self._lm = AutoModel.from_pretrained(lm_model)
        self._lm.eval()
        self._lm_device = lm_device
        self._lm.to(lm_device)
        self._no_space = no_space_lang
        self._emb_size = 768

    def collate_fn(self, batch):
        START = 0
        END = 2
        PAD = 1
        max_x = 0
        x_input = []
        x_lang = []
        y_output = []
        y_offset = []
        y_len = []
        x_text = []
        x_lang_word = []
        x_sent_len = []

        x_word_embeddings = []
        a_word_len = []
        for example in batch:
            for qq in ['prev', 'main', 'next']:
                sent = example[qq]
                toks, ids = self._tokenize(sent.text)
                for word in toks:
                    a_word_len.append(len(word))
                    x_lang_word.append(sent.lang_id)
        x_word_len = np.array(a_word_len, dtype=np.long)
        max_word_len = np.max(x_word_len)
        x_word_masks = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.float)
        x_word = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.long)
        x_word_case = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.long)
        c_word = 0
        for example in batch:
            sz = 0
            for qq in ['prev', 'main', 'next']:
                sent = example[qq]
                toks, ids = self._tokenize(sent.text)
                lst = toks
                sz += len(lst)
                for word in lst:
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
            x_sent_len.append(sz)

        for example in batch:
            current_sentence = example['main']
            prev_sentence = example['prev']
            next_sentence = example['next']
            x_lang.append(current_sentence.lang_id + 1)
            toks, ids = self._tokenize(prev_sentence.text)
            x_prev = ids
            toks, ids = self._tokenize(next_sentence.text)
            x_next = ids
            y_offset.append(len(x_prev) + 1)
            c_toks, ids = self._tokenize(current_sentence.text)
            x_main = ids
            y_len.append(len(x_main))
            x_len = len(x_prev) + len(x_main) + len(x_next)
            x_input.append([x_prev, x_main, x_next])
            x_text.append(c_toks)
            y_output.append(self._get_targets(current_sentence))
            if x_len > max_x:
                max_x = x_len

        x_out = np.ones((len(batch), max_x + 2), dtype=np.long) * PAD
        for ii in range(len(batch)):
            x_out[ii, 0] = START
            pos = 1
            x = x_input[ii][0]
            for jj in range(len(x)):
                x_out[ii, pos] = x[jj]
                pos += 1
            x = x_input[ii][1]
            for jj in range(len(x)):
                x_out[ii, pos] = x[jj]
                pos += 1
            x = x_input[ii][2]
            for jj in range(len(x)):
                x_out[ii, pos] = x[jj]
                pos += 1
            x_out[ii, pos] = END

        y_out = np.zeros((x_out.shape[0], x_out.shape[1]), dtype=np.long)
        for ii in range(x_out.shape[0]):
            for jj in range(y_len[ii]):
                index = y_offset[ii] + jj
                y_out[ii, index] = y_output[ii][jj]
        x_out = torch.tensor(x_out, device=self._lm_device)
        x_lang = torch.tensor(x_lang)
        y_out = torch.tensor(y_out)
        y_offset = torch.tensor(y_offset)
        y_len = torch.tensor(y_len)
        x_word = torch.tensor(x_word)
        x_word_case = torch.tensor(x_word_case)
        x_word_masks = torch.tensor(x_word_masks)
        x_word_len = torch.tensor(x_word_len)
        x_lang_word = torch.tensor(x_lang_word)
        x_sent_len = torch.tensor(x_sent_len)
        with torch.no_grad():
            x_out = self._lm(x_out)['last_hidden_state'].detach()
        return {'x_input': x_out, 'x_word_char': x_word, 'x_word_case': x_word_case, 'x_word_masks': x_word_masks,
                'x_word_len': x_word_len, 'x_word_lang': x_lang_word, 'x_text': x_text, 'x_lang': x_lang,
                'y_output': y_out, 'y_offset': y_offset, 'y_len': y_len, 'x_sent_len': x_sent_len}

    def _tokenize(self, text):
        if self._no_space:
            new_text = [ch for ch in text]
        else:
            import re
            punctuation = '''"’'()[]{}<>:,‒–—―…!.«»-?‘’“”;/⁄␠·&@*\\•^¤¢$€£¥₩₪†‡°¡¿¬#№%‰‱¶′§~¨_|¦⁂☞∴‽※"'''
            new_text = ''
            for ch in text:
                if re.match(u'[\u4e00-\u9fff]', ch):
                    new_text += ' ' + ch + ' '
                elif ch in punctuation:
                    new_text += ' ' + ch + ' '
                else:
                    new_text += ch

            tmp = new_text.replace('  ', ' ')
            while tmp != new_text:
                new_text = tmp
                tmp = new_text.replace('  ', ' ')

            new_text = new_text.split(' ')
        print("\n" + ("_" * 50))
        print(new_text)
        print("_" * 50)
        toks = self._tokenizer.tokenize(new_text, is_split_into_words=True)
        ids = self._tokenizer(new_text, is_split_into_words=True)['input_ids'][1:-1]
        r_toks = []
        r_ids = []
        if len(toks) != 0:  # empty text
            r_toks.append(toks[0])
            r_ids.append(ids[0])
        for ii in range(1, len(toks)):
            if toks[ii] != '▁':
                r_toks.append(toks[ii])
                r_ids.append(ids[ii])
        return r_toks, r_ids

    def _get_targets(self, sentence: Sentence):
        text = sentence.text
        toks, ids = self._tokenize(text)
        toks = [tok.replace('▁', '') for tok in toks]
        targets = [0 for _ in range(len(toks))]
        iToken = 0
        cl = 0
        for ii in range(len(targets)):
            target = 1  # nothing
            cl += len(toks[ii])
            if cl == len(sentence.tokens[iToken].text):
                iToken += 1
                cl = 0
                target = 2  # token
                if len(sentence.tokens[iToken].words) > 1:
                    target = 3  # multiword token
            if iToken == len(sentence.tokens):
                target = 4  # sentence end (+token)
            targets[ii] = target
        return targets

    def get_embeddings_size(self) -> int:
        return self._emb_size
