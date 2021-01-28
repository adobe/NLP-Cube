import sys
import random

sys.path.append('')
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModel
from cube3.io_utils.objects import Document, Sentence, Token, Word
from cube3.io_utils.encodings import Encodings

from collections import namedtuple


class TokenizationDataset(Dataset):
    def __init__(self, document: Document, shuffle=True):
        self._document = document
        self._shuffle = shuffle

    def __len__(self):
        return len(self._document.sentences)

    def __getitem__(self, item):
        # append two random sentences
        if self._shuffle:
            index1 = random.randint(0, len(self._document.sentences) - 1)
            index2 = random.randint(0, len(self._document.sentences) - 1)
        else:
            index1 = item - 1
            index2 = item + 1
        if index1 >= 0:
            prev = self._document.sentences[index1]
        else:
            prev = Sentence(sequence=[])
        if index2 < len(self._document.sentences):
            next = self._document.sentences[index2]
        else:
            next = Sentence(sequence=[])
        return {'main': self._document.sentences[item], 'prev': prev, 'next': next}


class MorphoDataset(Dataset):
    def __init__(self, document: Document):
        self._document = document

    def __len__(self):
        return len(self._document.sentences)

    def __getitem__(self, item):
        return self._document.sentences[item]


class LemmaDataset(Dataset):
    def __init__(self, document: Document):
        self._examples = []
        lookup = {}
        for sent in document.sentences:
            lang_id = sent.lang_id
            for w in sent.words:
                word = w.word
                lemma = w.lemma
                upos = w.upos

                key = (word, lang_id, upos)
                if key not in lookup:
                    lookup[key] = 1
                    example = {'word': word, 'upos': upos, 'lang_id': lang_id, 'target': lemma}
                    self._examples.append(example)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        return self._examples[item]


class CompoundDataset(Dataset):
    def __init__(self, document: Document):
        self._examples = []
        lookup = {}
        for sent in document:
            lang_id = sent.lang_id
            for t in sent.tokens:
                if len(t.words) > 1:
                    word = t.text
                    target = ' '.join([w.word for w in t.words])
                    key = (word, lang_id)
                    if key not in lookup:
                        lookup[key] = 1
                        example = {'word': word, 'lang_id': lang_id, 'target': target}
                        self._examples.append(example)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        return self._examples[item]


class Word2TargetCollate:
    def __init__(self, encodings: Encodings):
        self._encodings = encodings
        self._start_index = len(encodings.char2int)
        self._stop_index = len(encodings.char2int) + 1
        self._eol = len(encodings.char2int)

    def collate_fn(self, batch):
        max_input_len = max([len(e['word']) for e in batch])
        max_target_len = max([len(e['target']) for e in batch])
        n = len(batch)
        x_char = np.zeros((n, max_input_len + 2), dtype=np.long)
        x_case = np.zeros((n, max_input_len + 2), dtype=np.long)
        y_char = np.zeros((n, max_target_len + 1), dtype=np.long)
        y_case = np.zeros((n, max_target_len + 1), dtype=np.long)
        x_lang = np.zeros(n, dtype=np.long)
        x_upos = np.zeros(n, dtype=np.long)

        for ii in range(n):
            word = batch[ii]['word']
            target = batch[ii]['target']
            lang_id = batch[ii]['lang_id']
            upos = ''
            if 'upos' in batch[ii]:
                upos = batch[ii]['upos']
            x_lang[ii] = lang_id + 1
            if upos in self._encodings.upos2int:
                x_upos[ii] = self._encodings.upos2int[upos]
            else:
                x_upos[ii] = self._encodings.upos2int['<UNK>']

            i_char = 1
            x_char[ii, 0] = self._start_index
            for ch in word:
                if ch.lower() == ch.upper():
                    x_case[ii, i_char] = 1  # symbol
                elif ch.lower() != ch:
                    x_case[ii, i_char] = 2  # uppercase
                else:
                    x_case[ii, i_char] = 3  # lowercase
                if ch.lower() in self._encodings.char2int:
                    x_char[ii, i_char] = self._encodings.char2int[ch.lower()]
                else:
                    x_char[ii, i_char] = self._encodings.char2int['<UNK>']
                i_char += 1
            x_char[ii, i_char] = self._stop_index

            i_char = 0
            for ch in target:
                if ch.lower() == ch.upper():
                    y_case[ii, i_char] = 1  # symbol
                elif ch.lower() != ch:
                    y_case[ii, i_char] = 2  # uppercase
                else:
                    y_case[ii, i_char] = 3  # lowercase
                if ch.lower() in self._encodings.char2int:
                    y_char[ii, i_char] = self._encodings.char2int[ch.lower()]
                else:
                    y_char[ii, i_char] = self._encodings.char2int['<UNK>']
                i_char += 1
            y_char[ii, i_char] = self._eol

        rez = {
            'x_char': torch.tensor(x_char),
            'x_case': torch.tensor(x_case),
            'x_lang': torch.tensor(x_lang),
            'x_upos': torch.tensor(x_upos),
            'y_char': torch.tensor(y_char),
            'y_case': torch.tensor(y_case)
        }
        return rez


class TokenCollate:
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

    # def _normalize(self, text):
    #     import re
    #
    #     punctuation = '''"’'()[]{}<>:,‒–—―…!.«»-?‘’“”;/⁄␠·&@*\\•^¤¢$€£¥₩₪†‡°¡¿¬#№%‰‱¶′§~¨_|¦⁂☞∴‽※"'''
    #     new_text = ''
    #     for ch in text:
    #         if re.match(u'[\u4e00-\u9fff]', ch):
    #             new_text += ' ' + ch + ' '
    #         elif ch in punctuation:
    #             new_text += ' ' + ch + ' '
    #         else:
    #             new_text += ch
    #
    #     tmp = new_text.replace('  ', ' ')
    #     while tmp != new_text:
    #         new_text = tmp
    #         tmp = new_text.replace('  ', ' ')
    #     return new_text.strip()

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
            if iToken == len(sentence.tokens):
                target = 3  # sentence end (+token)
            targets[ii] = target
        return targets


class MorphoCollate:
    def __init__(self, encodings: Encodings, add_parsing=False):
        self._encodings = encodings
        self._add_parsing = add_parsing

    def collate_fn(self, batch: [Sentence]):
        a_sent_len = [len(sent.words) for sent in batch]
        a_word_len = []
        x_word_embeddings = []
        for sent in batch:
            for word in sent.words:
                a_word_len.append(len(word.word))
                x_word_embeddings.append(word._emb)
        x_sent_len = np.array(a_sent_len, dtype=np.long)
        x_word_len = np.array(a_word_len, dtype=np.long)
        max_sent_len = np.max(x_sent_len)
        max_word_len = np.max(x_word_len)
        x_sent_masks = np.zeros((len(batch), max_sent_len), dtype=np.float)
        x_word_masks = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.float)

        x_sent = np.zeros((len(batch), max_sent_len), dtype=np.long)
        x_word = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.long)
        x_word_case = np.zeros((x_word_len.shape[0], max_word_len), dtype=np.long)
        c_word = 0
        x_lang_sent = np.zeros((len(batch)), dtype=np.long)
        x_lang_word = []

        y_upos = np.zeros((x_sent.shape[0], x_sent.shape[1]), dtype=np.long)
        y_xpos = np.zeros((x_sent.shape[0], x_sent.shape[1]), dtype=np.long)
        y_attrs = np.zeros((x_sent.shape[0], x_sent.shape[1]), dtype=np.long)

        y_head = np.zeros((x_sent.shape[0], x_sent.shape[1]), dtype=np.long)
        y_label = np.zeros((x_sent.shape[0], x_sent.shape[1]), dtype=np.long)

        for iSent in range(len(batch)):
            sent = batch[iSent]
            x_lang_sent[iSent] = sent.lang_id + 1
            for iWord in range(len(sent.words)):
                word = sent.words[iWord]
                y_head[iSent, iWord] = word.head
                if word.label in self._encodings.label2int:
                    y_label[iSent, iWord] = self._encodings.label2int[word.label]
                if word.upos in self._encodings.upos2int:
                    y_upos[iSent, iWord] = self._encodings.upos2int[word.upos]
                if word.xpos in self._encodings.xpos2int:
                    y_xpos[iSent, iWord] = self._encodings.xpos2int[word.xpos]
                if word.attrs in self._encodings.attrs2int:
                    y_attrs[iSent, iWord] = self._encodings.attrs2int[word.attrs]
                word = sent.words[iWord].word
                x_sent_masks[iSent, iWord] = 1
                w = word.lower()
                x_lang_word.append(sent.lang_id + 1)
                if w in self._encodings.word2int:
                    x_sent[iSent, iWord] = self._encodings.word2int[w]
                else:
                    x_sent[iSent, iWord] = self._encodings.word2int['<UNK>']

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

        x_lang_word = np.array(x_lang_word)
        response = {
            'x_sent': torch.tensor(x_sent),
            'x_lang_sent': torch.tensor(x_lang_sent),
            'x_word': torch.tensor(x_word),
            'x_word_case': torch.tensor(x_word_case),
            'x_lang_word': torch.tensor(x_lang_word),
            'x_sent_len': torch.tensor(x_sent_len),
            'x_word_len': torch.tensor(x_word_len),
            'x_sent_masks': torch.tensor(x_sent_masks),
            'x_word_masks': torch.tensor(x_word_masks),
            'x_word_embeddings': torch.tensor(x_word_embeddings),
            'y_upos': torch.tensor(y_upos),
            'y_xpos': torch.tensor(y_xpos),
            'y_attrs': torch.tensor(y_attrs)
        }

        if self._add_parsing:
            response['y_head'] = torch.tensor(y_head)
            response['y_label'] = torch.tensor(y_label)

        return response


class LMHelper:
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
        input_ids = np.ones((len(new_sents), max_len), dtype=np.long) * PAD  # pad everything
        for ii in range(input_ids.shape[0]):
            for jj in range(input_ids.shape[1]):
                if jj < len(new_sents[ii]):
                    input_ids[ii, jj] = new_sents[ii][jj]
        with torch.no_grad():
            out = self._xlmr(torch.tensor(input_ids, device=self._device), return_dict=True)
            we = out['last_hidden_state'].detach().cpu().numpy()

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
                    m = np.zeros((768), dtype=np.float)
                word_emb.append(m)
        # word_emb = torch.cat(word_emb, dim=0)

        return word_emb

    def apply(self, doc: Document):
        import tqdm
        for sent in tqdm.tqdm(doc.sentences, desc="Pre-computing embeddings", unit="sent"):
            wemb = self._compute_we([sent])
            for ii in range(len(wemb)):
                sent.words[ii]._emb = wemb[ii]


Arc = namedtuple('Arc', ('tail', 'weight', 'head'))


class GreedyDecoder:
    def _valid(self, arc, tree):
        # just one head
        for sa in tree:
            if sa.tail == arc.tail:
                return False
        stack = [arc.head]
        pos = 0
        used = [False] * len(tree)
        while pos < len(stack):
            for zz in range(len(tree)):
                if tree[zz].tail == stack[pos] and not used[zz]:
                    used[zz] = True
                    stack.append(tree[zz].head)
                    if tree[zz].head == arc.tail:
                        return False
            pos += 1
            # print pos,len(stack)
        return True

    def _get_sort_key(self, item):
        return item.weight

    def _greedy_tree(self, arcs):
        arcs = sorted(arcs, key=self._get_sort_key, reverse=True)
        # print arcs
        final_tree = []
        for index in range(len(arcs)):
            if self._valid(arcs[index], final_tree):
                final_tree.append(arcs[index])
                # print arcs[index]
        return final_tree

    def _make_ordered_list(self, tree, nWords):
        lst = [0] * nWords  # np.zeros(nWords)
        for arc in tree:
            # arc = tree[index]
            tail = arc.tail
            head = arc.head
            lst[tail] = head
        return lst[1:]

    def decode(self, score, lens):
        best_tree_list = []
        for ii in range(score.shape[0]):
            # norm_score = score[ii, :lens[ii], :lens[ii]]
            norm_score = np.zeros((lens[ii] + 1, lens[ii] + 1))
            for wii in range(lens[ii]):
                for wjj in range(lens[ii] + 1):
                    norm_score[wii + 1, wjj] = score[ii, wii, wjj]
            nWords = norm_score.shape[0]  # len(norm_score)
            g = []
            for iSrc in range(1, nWords):
                for iDst in range(1, nWords):
                    if iDst != iSrc:
                        a = Arc(iSrc, norm_score[iSrc][iDst], iDst)
                        g.append(a)
            tree = self._greedy_tree(g)
            best_tree = self._make_ordered_list(tree, nWords)
            best_tree_list.append(best_tree)
        return best_tree_list


# Code adapted from https://github.com/tdozat/Parser-v3/blob/master/scripts/chuliu_edmonds.py
class ChuLiuEdmondsDecoder:
    def __init__(self):
        pass

    def _tarjan(self, tree):
        indices = -np.ones_like(tree)
        lowlinks = -np.ones_like(tree)
        onstack = np.zeros_like(tree, dtype=bool)
        stack = list()
        _index = [0]
        cycles = []

        def strong_connect(i):
            _index[0] += 1
            index = _index[-1]
            indices[i] = lowlinks[i] = index - 1
            stack.append(i)
            onstack[i] = True
            dependents = np.where(np.equal(tree, i))[0]
            for j in dependents:
                if indices[j] == -1:
                    strong_connect(j)
                    lowlinks[i] = min(lowlinks[i], lowlinks[j])
                elif onstack[j]:
                    lowlinks[i] = min(lowlinks[i], indices[j])

            # There's a cycle!
            if lowlinks[i] == indices[i]:
                cycle = np.zeros_like(indices, dtype=bool)
                while stack[-1] != i:
                    j = stack.pop()
                    onstack[j] = False
                    cycle[j] = True
                stack.pop()
                onstack[i] = False
                cycle[i] = True
                if cycle.sum() > 1:
                    cycles.append(cycle)
            return

        for i in range(len(tree)):
            if indices[i] == -1:
                strong_connect(i)
        return cycles

    def _chuliu_edmonds(self, scores):
        scores *= (1 - np.eye(scores.shape[0]))
        scores[0] = 0
        scores[0, 0] = 1
        tree = np.argmax(scores, axis=1)
        cycles = self._tarjan(tree)
        if not cycles:
            return tree
        else:
            # t = len(tree); c = len(cycle); n = len(noncycle)
            # locations of cycle; (t) in [0,1]
            cycle = cycles.pop()
            # indices of cycle in original tree; (c) in t
            cycle_locs = np.where(cycle)[0]
            # heads of cycle in original tree; (c) in t
            cycle_subtree = tree[cycle]
            # scores of cycle in original tree; (c) in R
            cycle_scores = scores[cycle, cycle_subtree]
            # total score of cycle; () in R
            cycle_score = cycle_scores.prod()

            # locations of noncycle; (t) in [0,1]
            noncycle = np.logical_not(cycle)
            # indices of noncycle in original tree; (n) in t
            noncycle_locs = np.where(noncycle)[0]
            # print(cycle_locs, noncycle_locs)

            # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
            metanode_head_scores = scores[cycle][:, noncycle] / cycle_scores[:, None] * cycle_score
            # scores of cycle's potential dependents; (n x c) in R
            metanode_dep_scores = scores[noncycle][:, cycle]
            # best noncycle head for each cycle dependent; (n) in c
            metanode_heads = np.argmax(metanode_head_scores, axis=0)
            # best cycle head for each noncycle dependent; (n) in c
            metanode_deps = np.argmax(metanode_dep_scores, axis=1)

            # scores of noncycle graph; (n x n) in R
            subscores = scores[noncycle][:, noncycle]
            # pad to contracted graph; (n+1 x n+1) in R
            subscores = np.pad(subscores, ((0, 1), (0, 1)), 'constant')
            # set the contracted graph scores of cycle's potential heads; (c x n)[:, (n) in n] in R -> (n) in R
            subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
            # set the contracted graph scores of cycle's potential dependents; (n x c)[(n) in n] in R-> (n) in R
            subscores[:-1, -1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]

            # MST with contraction; (n+1) in n+1
            contracted_tree = self._chuliu_edmonds(subscores)
            # head of the cycle; () in n
            # print(contracted_tree)
            cycle_head = contracted_tree[-1]
            # fixed tree: (n) in n+1
            contracted_tree = contracted_tree[:-1]
            # initialize new tree; (t) in 0
            new_tree = -np.ones_like(tree)
            # print(0, new_tree)
            # fixed tree with no heads coming from the cycle: (n) in [0,1]
            contracted_subtree = contracted_tree < len(contracted_tree)
            # add the nodes to the new tree (t)[(n)[(n) in [0,1]] in t] in t = (n)[(n)[(n) in [0,1]] in n] in t
            new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[contracted_tree[contracted_subtree]]
            # print(1, new_tree)
            # fixed tree with heads coming from the cycle: (n) in [0,1]
            contracted_subtree = np.logical_not(contracted_subtree)
            # add the nodes to the tree (t)[(n)[(n) in [0,1]] in t] in t = (c)[(n)[(n) in [0,1]] in c] in t
            new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
            # print(2, new_tree)
            # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
            new_tree[cycle_locs] = tree[cycle_locs]
            # print(3, new_tree)
            # root of the cycle; (n)[() in n] in c = () in c
            cycle_root = metanode_heads[cycle_head]
            # add the root of the cycle to the new tree; (t)[(c)[() in c] in t] = (c)[() in c]
            new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
            # print(4, new_tree)
            return new_tree

    def _chuliu_edmonds_one_root(self, scores):
        """"""

        scores = scores.astype(np.float64)
        tree = self._chuliu_edmonds(scores)
        roots_to_try = np.where(np.equal(tree[1:], 0))[0] + 1
        if len(roots_to_try) == 1:
            return tree

        # Look at all roots that are more likely than we would expect
        if len(roots_to_try) == 0:
            roots_to_try = np.where(scores[1:, 0] >= 1 / len(scores))[0] + 1
        # *sigh* just grab the most likely one
        if len(roots_to_try) == 0:
            roots_to_try = np.array([np.argmax(scores[1:, 0]) + 1])

        # -------------------------------------------------------------
        def set_root(scores, root):
            root_score = scores[root, 0]
            scores = np.array(scores)
            scores[1:, 0] = 0
            scores[root] = 0
            scores[root, 0] = 1
            return scores, root_score

        # -------------------------------------------------------------

        best_score, best_tree = -np.inf, None  # This is what's causing it to crash
        for root in roots_to_try:
            _scores, root_score = set_root(scores, root)
            _tree = self._chuliu_edmonds(_scores)
            tree_probs = _scores[np.arange(len(_scores)), _tree]
            tree_score = np.log(tree_probs).sum() + np.log(root_score) if tree_probs.all() else -np.inf
            if tree_score > best_score:
                best_score = tree_score
                best_tree = _tree
        try:
            assert best_tree is not None
        except:
            with open('debug.log', 'w') as f:
                f.write('{}: {}, {}\n'.format(tree, scores, roots_to_try))
                f.write('{}: {}, {}, {}\n'.format(_tree, _scores, tree_probs, tree_score))
            raise
        return best_tree

    def decode(self, score, lens):
        best_tree_list = []
        for ii in range(score.shape[0]):
            # norm_score = score[ii, :lens[ii], :lens[ii]]
            norm_score = np.zeros((lens[ii] + 1, lens[ii] + 1))
            for wii in range(lens[ii]):
                for wjj in range(lens[ii] + 1):
                    norm_score[wii + 1, wjj] = score[ii, wii, wjj]
            nWords = norm_score.shape[0]  # len(norm_score)
            norm_score *= (1 - np.eye(nWords))
            tree = self._chuliu_edmonds_one_root(norm_score)
            best_tree_list.append(tree[1:])
        return best_tree_list
