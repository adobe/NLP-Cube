import sys
import random
from abc import abstractmethod

sys.path.append('')
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from cube.io_utils.objects import Document, Sentence, Token, Word
from cube.io_utils.encodings import Encodings

from collections import namedtuple
from transformers import AutoModel, AutoTokenizer


def unpack(data: torch.Tensor, sizes: [], max_size: int, device: str):
    pos = 0
    blist = []
    for ii in range(len(sizes)):
        slist = []
        for jj in range(sizes[ii]):
            slist.append(data[pos, :].unsqueeze(0))
            pos += 1

        for jj in range(max_size - sizes[ii]):
            slist.append(torch.zeros((1, data.shape[-1]), device=device, dtype=torch.float))
        slist = torch.cat(slist, dim=0)
        blist.append(slist.unsqueeze(0))

    blist = torch.cat(blist, dim=0)
    return blist


def mask_concat(representations, drop_prob: float, training: bool, device: str):
    if training:
        masks = []
        for ii in range(len(representations)):
            mask = np.ones((representations[ii].shape[0], representations[ii].shape[1]), dtype=np.long)
            masks.append(mask)

        for ii in range(masks[0].shape[0]):
            for jj in range(masks[0].shape[1]):
                mult = 1
                for kk in range(len(masks)):
                    p = random.random()
                    if p < drop_prob:
                        mult += 1
                        masks[kk][ii, jj] = 0
                for kk in range(len(masks)):
                    masks[kk][ii, jj] *= mult
        for kk in range(len(masks)):
            masks[kk] = torch.tensor(masks[kk], device=device)

        for kk in range(len(masks)):
            representations[kk] = representations[kk] * masks[kk].unsqueeze(2)

    return torch.cat(representations, dim=-1)


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
    def __init__(self, document: Document, for_training=True):
        self._examples = []
        lookup = {}
        for sent in document.sentences:
            lang_id = sent.lang_id
            for w in sent.words:
                word = w.word
                lemma = w.lemma
                upos = w.upos

                key = (word, lang_id, upos)
                if key not in lookup or for_training is False:
                    lookup[key] = 1
                    example = {'word': word, 'upos': upos, 'lang_id': lang_id, 'target': lemma}
                    self._examples.append(example)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        return self._examples[item]


class CompoundDataset(Dataset):
    def __init__(self, document: Document, for_training=True):
        self._examples = []
        lookup = {}
        for sent in document.sentences:
            lang_id = sent.lang_id
            if for_training is True:
                for t in sent.tokens:
                    if len(t.words) > 1:
                        word = t.text
                        target = ' '.join([w.word for w in t.words])
                        key = (word, lang_id)
                        # if key not in lookup:
                        lookup[key] = 1
                        example = {'word': word, 'lang_id': lang_id, 'target': target}
                        self._examples.append(example)
            else:
                for t in sent.tokens:
                    example = {'word': t.words[0].word, 'lang_id': lang_id, 'target': ""}
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


class MorphoCollate:
    def __init__(self, encodings: Encodings, add_parsing=False, rhl_win_size=7):
        self._encodings = encodings
        self._add_parsing = add_parsing
        self._rhl_win_size = rhl_win_size

    def collate_fn(self, batch: [Sentence]):
        a_sent_len = [len(sent.words) for sent in batch]
        a_word_len = []

        x_word_embeddings = [[] for _ in range(len(batch[0].words[0].emb))]
        for sent in batch:
            for word in sent.words:
                a_word_len.append(len(word.word))
                for ii in range(len(word.emb)):
                    x_word_embeddings[ii].append(word.emb[ii])
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
        y_rhl = np.zeros((x_sent.shape[0], x_sent.shape[1]), dtype=np.long)

        for iSent in range(len(batch)):
            sent = batch[iSent]
            x_lang_sent[iSent] = sent.lang_id + 1
            for iWord in range(len(sent.words)):
                word = sent.words[iWord]
                y_head[iSent, iWord] = word.head
                rhl = word.head - iWord + self._rhl_win_size
                rhl = np.clip(rhl, 0, self._rhl_win_size * 2 - 1)
                y_rhl[iSent, iWord] = rhl

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
        x_word_embeddings = [torch.tensor(t) for t in x_word_embeddings]
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
            'x_word_embeddings': x_word_embeddings,
            'y_upos': torch.tensor(y_upos),
            'y_xpos': torch.tensor(y_xpos),
            'y_attrs': torch.tensor(y_attrs)
        }

        if self._add_parsing:
            response['y_head'] = torch.tensor(y_head)
            response['y_label'] = torch.tensor(y_label)
            response['y_rhl'] = torch.tensor(y_rhl)

        return response


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
