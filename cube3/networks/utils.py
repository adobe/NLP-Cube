import sys

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


class MorphoDataset(Dataset):
    def __init__(self, document: Document):
        self._document = document

    def __len__(self):
        return len(self._document.sentences)

    def __getitem__(self, item):
        return self._document.sentences[item]


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
        END = 1
        PAD = 2
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
        input_ids = np.ones((len(new_sents), max_len), dtype=np.long) * 2  # pad everything
        for ii in range(input_ids.shape[0]):
            for jj in range(input_ids.shape[1]):
                if jj < len(new_sents[ii]):
                    input_ids[ii, jj] = new_sents[ii][jj]
        with torch.no_grad():
            we = self._xlmr(torch.tensor(input_ids, device=self._device))['last_hidden_state'].detach().cpu().numpy()

        word_emb = []
        for ii in range(len(batch)):
            for jj in range(len(batch[ii].words)):
                pieces = word2pieces[ii, jj]
                m = we[pieces[0][0], pieces[0][1]]
                for zz in range(len(pieces) - 1):
                    m += we[pieces[zz][0], pieces[zz][1]]
                m = m / len(pieces)
                word_emb.append(m)
        # word_emb = torch.cat(word_emb, dim=0)

        return word_emb

    def apply(self, doc: Document):
        import tqdm
        for sent in tqdm.tqdm(doc.sentences):
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
