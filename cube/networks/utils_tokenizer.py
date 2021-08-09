import sys
import torch
import numpy as np

sys.path.append('')
from typing import *
from abc import abstractmethod
from transformers import AutoModel, AutoTokenizer
from cube.io_utils.encodings import Encodings
from cube.io_utils.objects import Sentence
from cube.networks.lm import LMHelperLanguasito, LMHelperFT
from torch.utils.data.dataset import Dataset


class TokenCollate:
    def __init__(self):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    @abstractmethod
    def get_embeddings_size(self) -> int:
        pass

    @abstractmethod
    def collate_fn_live(self, text, lang_id: int, batch_size: int):
        pass


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


def _make_example_from_raw(toks, iBatch, seq_len, overlap):
    batch = []
    num_batches = len(toks[0]) // seq_len
    if len(toks[0]) % seq_len != 0:
        num_batches += 1
    start = iBatch * seq_len
    stop = min(iBatch * seq_len + seq_len, len(toks[0]))
    current = toks[0][start:stop]
    left = max(0, start - overlap)
    right = min(len(toks[0]), stop + overlap)
    prev = toks[0][left:start]
    next = toks[0][stop:right]
    if len(toks) == 2:  # TF or languasito
        current_spa = toks[1][start:stop]
        prev_spa = toks[1][left:start]
        next_spa = toks[1][stop:right]
        prev = (prev, prev_spa)
        main = (current, current_spa)
        next = (next, next_spa)
    else:
        current_spa = toks[2][start:stop]
        prev_spa = toks[2][left:start]
        next_spa = toks[2][stop:right]
        current_ids = toks[1][start:stop]
        prev_ids = toks[1][left:start]
        next_ids = toks[1][stop:right]
        prev = (prev, prev_ids, prev_spa)
        main = (current, current_ids, current_spa)
        next = (next, next_ids, next_spa)

    # if len(prev)==0:
    #    prev=['']
    # if len(next)==0:
    #    next=['']
    example = {'prev': prev, 'main': main, 'next': next}
    return example


class TokenDatasetLive(Dataset):
    def __init__(self, raw_text, pretokenize_func, seq_len=300, overlap=100):
        self._tokenize = pretokenize_func
        if raw_text[-1] != ' ':
            raw_text += ' '
        self._toks = pretokenize_func(raw_text)  # self._tokenize(raw_text)
        self._seq_len = seq_len
        self._overlap = overlap
        self._num_examples = len(self._toks[0]) // seq_len
        if len(self._toks[0]) % seq_len != 0:
            self._num_examples += 1

    def __len__(self):
        return self._num_examples

    def __getitem__(self, item):
        return _make_example_from_raw(self._toks, item, self._seq_len, self._overlap)


class TokenCollateFTLanguasito(TokenCollate):
    def __init__(self, encodings: Encodings, lm_model: str = None, lm_device: str = 'cuda:0', no_space_lang=False,
                 lang_id=None):
        self._encodings = encodings
        # from languasito.utils import LanguasitoTokenizer
        self._tokenizer = LanguasitoTokenizer(no_space_language=no_space_lang)
        self._emb_size = 0
        self._lm_model = lm_model
        self._lm_device = lm_device
        self._lang_id = lang_id
        self.max_seq_len = -1
        parts = lm_model.split(':')
        if parts[0] == 'fasttext':
            self._lm_helper = LMHelperFT(device=lm_device, model=parts[1])
            self._emb_size = [300]
        elif parts[0] == 'languasito':
            self._lm_helper = LMHelperLanguasito(device=lm_device, model=parts[1])
            self._emb_size = [512]
        else:
            print("UserWarning: unsupported LM type for tokenizer")

    def collate_fn(self, batch):
        START = 0
        END = 2
        PAD = 1
        max_x = 0
        x_input = []
        x_input_spa = []
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
                if self._lang_id is None:
                    toks = self._tokenizer(sent.text)
                    l_id = sent.lang_id
                else:
                    toks, spa = sent
                    l_id = self._lang_id
                for word in toks:
                    a_word_len.append(len(word))
                    x_lang_word.append(l_id)

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
                if self._lang_id is None:
                    toks = self._tokenizer(sent.text)
                else:
                    toks, spa = sent
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
            if self._lang_id is None:
                x_lang.append(current_sentence.lang_id + 1)
            else:
                x_lang.append(self._lang_id + 1)
            # toks, ids = self._tokenize(prev_sentence.text)
            if self._lang_id is None:
                toks, spa = self.get_tokens(prev_sentence.text)
            else:
                toks, spa = prev_sentence
            x_prev = toks
            x_prev_spa = spa
            # toks, ids = self._tokenize(next_sentence.text)
            if self._lang_id is None:
                toks, spa = self.get_tokens(next_sentence.text)
            else:
                toks, spa = next_sentence
            x_next = toks
            x_next_spa = spa
            y_offset.append(len(x_prev))
            # c_toks, ids = self._tokenize(current_sentence.text)
            if self._lang_id is None:
                c_toks, c_spa = self.get_tokens(current_sentence.text)
            else:
                c_toks, c_spa = current_sentence
            x_main = c_toks
            x_main_spa = c_spa
            y_len.append(len(x_main))
            x_len = len(x_prev) + len(x_main) + len(x_next)
            x_input.append([x_prev, x_main, x_next])
            x_input_spa.append([x_prev_spa, x_main_spa, x_next_spa])
            x_text.append(c_toks)
            if self._lang_id is None:
                y_output.append(self._get_targets(current_sentence))
            else:
                y_output.append(np.zeros(len(c_toks)))

            if x_len > max_x:
                max_x = x_len

        x_for_emb = []
        for example in x_input:
            toks = example[0] + example[1] + example[2]
            x_for_emb.append(toks)

        x_emb = self._lm_helper.apply_raw(x_for_emb)
        max_len = max([len(x) for x in x_emb])
        x_out = np.zeros((len(x_emb), max_len, self._emb_size[0]), dtype=np.float)
        x_out_spa = np.zeros((len(x_emb), max_len), dtype=np.long)
        for ii in range(x_out.shape[0]):
            for jj in range(x_out.shape[1]):
                if jj < len(x_emb[ii]):
                    x_out[ii, jj, :] = x_emb[ii][jj]

            pos = 0
            spa = x_input_spa[ii][0]
            for jj in range(len(spa)):
                x_out_spa[ii, pos] = spa[jj]
                pos += 1
            spa = x_input_spa[ii][1]
            for jj in range(len(spa)):
                x_out_spa[ii, pos] = spa[jj]
                pos += 1
            spa = x_input_spa[ii][2]
            for jj in range(len(spa)):
                x_out_spa[ii, pos] = spa[jj]
                pos += 1

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
        x_input_spa = torch.tensor(x_out_spa)

        return {'x_input': [x_out], 'x_input_spa': x_input_spa, 'x_word_char': x_word, 'x_word_case': x_word_case,
                'x_word_masks': x_word_masks, 'x_word_len': x_word_len, 'x_word_lang': x_lang_word, 'x_text': x_text,
                'x_lang': x_lang, 'y_output': y_out, 'y_offset': y_offset, 'y_len': y_len, 'x_sent_len': x_sent_len}

    def _get_targets(self, sentence: Sentence):
        text = sentence.text
        toks = self._tokenizer(text)

        targets = [0 for _ in range(len(toks))]
        iToken = 0
        cl = 0
        for ii in range(len(targets)):
            target = 1  # nothing
            cl += len(toks[ii].replace(' ', ''))
            if cl == len(sentence.tokens[iToken].text.replace(' ', '')):
                iToken += 1
                cl = 0
                target = 2  # token
                if len(sentence.tokens[iToken - 1].words) > 1:
                    target = 3  # multiword token
                    print(target)
            if iToken == len(sentence.tokens):
                target = 4  # sentence end (+token)
                for tt in range(ii, len(targets)):
                    targets[ii] = target
                break
            targets[ii] = target
        return targets

    def get_tokens(self, text):
        toks = self._tokenizer(text)
        spa = [0 for _ in range(len(toks))]
        t_pos = 0
        for ii in range(len(toks)):
            t_pos += len(toks[ii])
            if t_pos < len(text) and text[t_pos] == ' ':
                spa[ii] = 2
                while t_pos < len(text) and text[t_pos] == ' ':
                    t_pos += 1
            else:
                spa[ii] = 1
        return toks, spa

    def get_embeddings_size(self) -> List[int]:
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
            self._emb_size = [300]
        elif parts[0] == 'languasito':
            self._lm_helper = LMHelperLanguasito(device=self._lm_device, model=parts[1])
            self._emb_size = [512]


class TokenCollateHF(TokenCollate):
    def __init__(self, encodings: Encodings, lm_device, lm_model=None, no_space_lang=False, lang_id=None):
        if lm_model is None:
            lm_model = 'xlm-roberta-base'
        self._encodings = encodings  # this is currently not used - we keep it for future development
        self._pretokenizer = LanguasitoTokenizer(no_space_language=no_space_lang)
        self._tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self._lm = AutoModel.from_pretrained(lm_model, output_hidden_states=True)
        self._lm.eval()
        self._lm_device = lm_device
        self._lm.to(lm_device)
        self._no_space = no_space_lang
        self._emb_size = [768 for _ in range(13)]
        self._lang_id = lang_id

        self.max_seq_len = self._tokenizer.model_max_length

    def get_tokens(self, text):
        toks, ids = self._tokenize(text)
        spa = [0 for _ in range(len(toks))]
        t_pos = 0
        for ii in range(len(toks)):
            t_pos += len(toks[ii].replace('▁', ''))
            if t_pos < len(text) and text[t_pos] == ' ':
                spa[ii] = 2
                while t_pos < len(text) and text[t_pos] == ' ':
                    t_pos += 1
            else:
                spa[ii] = 1
        return toks, ids, spa

    def collate_fn(self, batch):
        START = 0
        END = 2
        PAD = 1
        max_x = 0
        x_input = []
        x_input_spa = []
        x_lang = []
        y_output = []
        y_offset = []
        y_len = []
        x_text = []
        x_spa = []
        x_lang_word = []
        x_sent_len = []

        x_word_embeddings = []
        a_word_len = []
        for example in batch:
            for qq in ['prev', 'main', 'next']:
                sent = example[qq]
                if self._lang_id is None:
                    l_id = sent.lang_id
                    text = sent.text
                    toks, ids, spa = self.get_tokens(text)
                else:
                    l_id = self._lang_id
                    if len(sent) == 3:
                        toks, ids, spa = sent
                    else:
                        toks, ids, spa = [], [], []
                for word in toks:
                    a_word_len.append(len(word))
                    x_lang_word.append(l_id)
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
                if self._lang_id is None:
                    l_id = sent.lang_id
                    toks, ids, spa = self.get_tokens(sent.text)
                else:
                    l_id = self._lang_id
                    if len(sent) == 3:
                        toks, ids, spa = sent
                    else:
                        toks, ids, spa = [], [], []

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
            if self._lang_id is None:
                x_lang.append(current_sentence.lang_id + 1)
            else:
                x_lang.append(self._lang_id + 1)
            if self._lang_id is None:
                toks, ids, spa = self.get_tokens(prev_sentence.text)
            else:
                if len(prev_sentence) == 3:
                    toks, ids, spa = prev_sentence
                else:
                    toks, ids, spa = [], [], []
            x_prev = ids
            x_prev_spa = spa
            if self._lang_id is None:
                toks, ids, spa = self.get_tokens(next_sentence.text)
            else:
                if len(next_sentence) == 3:
                    toks, ids, spa = next_sentence
                else:
                    toks, ids, spa = [], [], []
            x_next = ids
            x_next_spa = spa
            y_offset.append(len(x_prev))
            if self._lang_id is None:
                c_toks, ids, c_spa = self.get_tokens(current_sentence.text)
            else:
                if len(current_sentence) == 3:
                    c_toks, ids, c_spa = current_sentence
                else:
                    c_toks, ids, c_spa = [], [], []

            x_main = ids
            x_main_spa = c_spa
            y_len.append(len(x_main))
            x_len = len(x_prev) + len(x_main) + len(x_next)
            x_input.append([x_prev, x_main, x_next])
            x_input_spa.append([x_prev_spa, x_main_spa, x_next_spa])
            x_text.append(c_toks)
            x_spa.append(c_spa)
            if self._lang_id is None:
                y_output.append(self._get_targets(current_sentence))
            else:
                y_output.append([0 for _ in range(len(c_toks))])
            if x_len > max_x:
                max_x = x_len

        x_out = np.ones((len(batch), max_x), dtype=np.long) * PAD
        x_out_spa = np.zeros((len(batch), max_x), dtype=np.long)
        for ii in range(len(batch)):
            # x_out[ii, 0] = START
            pos = 0
            x = x_input[ii][0]
            x_spa = x_input_spa[ii][0]
            for jj in range(len(x)):
                x_out[ii, pos] = x[jj]
                x_out_spa[ii, pos] = x_spa[jj]
                pos += 1
            x = x_input[ii][1]
            x_spa = x_input_spa[ii][1]
            for jj in range(len(x)):
                x_out[ii, pos] = x[jj]
                x_out_spa[ii, pos] = x_spa[jj]
                pos += 1
            x = x_input[ii][2]
            x_spa = x_input_spa[ii][2]
            for jj in range(len(x)):
                x_out[ii, pos] = x[jj]
                x_out_spa[ii, pos] = x_spa[jj]
                pos += 1
            # x_out[ii, pos] = END

        y_out = np.zeros((x_out.shape[0], x_out.shape[1]), dtype=np.long)
        for ii in range(x_out.shape[0]):
            for jj in range(y_len[ii]):
                index = y_offset[ii] + jj
                y_out[ii, index] = y_output[ii][jj]
        x_out = torch.tensor(x_out, device=self._lm_device)
        x_lang = torch.tensor(x_lang)
        y_out = torch.tensor(y_out)
        x_input_spa = torch.tensor(x_out_spa, dtype=torch.long)
        y_offset = torch.tensor(y_offset)
        y_len = torch.tensor(y_len)
        x_word = torch.tensor(x_word)
        x_word_case = torch.tensor(x_word_case)
        x_word_masks = torch.tensor(x_word_masks)
        x_word_len = torch.tensor(x_word_len)
        x_lang_word = torch.tensor(x_lang_word)
        x_sent_len = torch.tensor(x_sent_len)
        with torch.no_grad():
            if x_out.size()[1] > self.max_seq_len:
                # print()
                # print(x_out.size())
                # hack to skip batch if len is to big
                # we cannot return none because pytorch lightning will complain, so we set x_input = None and check
                # it in the train_step. TODO check long string in processing input
                return {'x_input': None, 'x_input_spa': x_input_spa, 'x_word_char': x_word, 'x_word_case': x_word_case,
                        'x_word_masks': x_word_masks,
                        'x_word_len': x_word_len, 'x_word_lang': x_lang_word, 'x_text': x_text, 'x_lang': x_lang,
                        'y_output': y_out, 'y_offset': y_offset, 'y_len': y_len, 'x_sent_len': x_sent_len}
            x_out = self._lm(x_out)['hidden_states']
            x_out = [t.detach() for t in x_out]
        return {'x_input': x_out, 'x_input_spa': x_input_spa, 'x_word_char': x_word, 'x_word_case': x_word_case,
                'x_word_masks': x_word_masks,
                'x_word_len': x_word_len, 'x_word_lang': x_lang_word, 'x_text': x_text, 'x_lang': x_lang,
                'y_output': y_out, 'y_offset': y_offset, 'y_len': y_len, 'x_sent_len': x_sent_len}

    def _tokenize(self, text):
        if self._no_space:
            new_text = [ch for ch in text]
        else:
            new_text = self._pretokenizer(text)
        # print("\n" + ("_" * 50))
        # print(new_text)
        # print("_" * 50)
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
        # if len(r_toks) > 509 or len(r_ids) > 509 or len(r_toks) < 1 or len(r_ids) < 1:
        #    print(f"\n>> text:[{text}] [{len(r_toks)}] [{len(r_ids)}]")
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
                if len(sentence.tokens[iToken - 1].words) > 1:
                    target = 3  # multiword token
            if iToken == len(sentence.tokens):
                target = 4  # sentence end (+token)
                for tt in range(ii, len(targets)):
                    targets[ii] = target
                break
            targets[ii] = target
        return targets

    def get_embeddings_size(self) -> List[int]:
        return self._emb_size
