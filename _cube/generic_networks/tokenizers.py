#
# Authors: Tiberiu Boros, Stefan Daniel Dumitrescu
#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import dynet as dy
import numpy as np
import os
import random
import string
from cube.misc.misc import fopen

from cube.misc.misc import get_eta, pretty_time, log_progress, line_count
from cube.io_utils.conll import ConllEntry
from cube.generic_networks.crf import CRFDecoder


class DummyTokenizer:
    def __init__(self, config, encodings, num_languages=1, dict=None, runtime=False):
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.encodings = encodings
        self.config = config
        self.dict = dict
        self.char_lookup = self.model.add_lookup_parameters((len(self.encodings.char2int), config.char_emb_size))
        self.case_lookup = self.model.add_lookup_parameters((3, 32))
        self.lang_lookup = self.model.add_lookup_parameters((num_languages, config.lang_emb_size))

        self.LAYER_SIZE = 300
        self.NUM_LAYERS = 15
        self.WINDOW_SIZE = 2
        inp_size = config.lang_emb_size + config.lang_emb_size + 32
        if dict is not None:
            inp_size += 1
        self._proj_w = self.model.add_parameters((self.LAYER_SIZE, inp_size))
        self._proj_b = self.model.add_parameters((self.LAYER_SIZE))

        self._fw_gate_w = []
        self._fw_gate_b = []
        self._fw_act_w = []
        self._fw_act_b = []
        self._fw_skip_w = []
        self._fw_skip_b = []
        self._bw_gate_w = []
        self._bw_gate_b = []
        self._bw_act_w = []
        self._bw_act_b = []
        self._bw_skip_w = []
        self._bw_skip_b = []
        inp_size = self.LAYER_SIZE * (self.WINDOW_SIZE + 1)
        for ii in range(self.NUM_LAYERS):
            self._fw_gate_w.append(self.model.add_parameters((self.LAYER_SIZE, inp_size)))
            self._fw_gate_b.append(self.model.add_parameters((self.LAYER_SIZE)))
            self._fw_act_w.append(self.model.add_parameters((self.LAYER_SIZE, inp_size)))
            self._fw_act_b.append(self.model.add_parameters((self.LAYER_SIZE)))
            self._fw_skip_w.append(self.model.add_parameters((self.LAYER_SIZE, inp_size)))
            self._fw_skip_b.append(self.model.add_parameters((self.LAYER_SIZE)))
            self._bw_gate_w.append(self.model.add_parameters((self.LAYER_SIZE, inp_size)))
            self._bw_gate_b.append(self.model.add_parameters((self.LAYER_SIZE)))
            self._bw_act_w.append(self.model.add_parameters((self.LAYER_SIZE, inp_size)))
            self._bw_act_b.append(self.model.add_parameters((self.LAYER_SIZE)))
            self._bw_skip_w.append(self.model.add_parameters((self.LAYER_SIZE, inp_size)))
            self._bw_skip_b.append(self.model.add_parameters((self.LAYER_SIZE)))
            inp_size = self.LAYER_SIZE * (self.WINDOW_SIZE + 1)

        self.label2int = {'B': 0, 'I': 1, 'E': 2, 'S': 3, 'X': 4, 'BM': 5, 'IM': 6, 'EM': 7, 'SM': 8, 'T': 9, 'U': 10,
                          'UM': 11}
        self.label_list = ['B', 'I', 'E', 'S', 'X', 'BM', 'IM', 'EM', 'SM', 'T', 'U', 'UM']
        # self.crf = CRFDecoder(self.model, self.LAYER_SIZE, 100, len(self.label_list))
        self.aux1_softmax_output_w = self.model.add_parameters((len(self.label_list), self.LAYER_SIZE))
        self.aux1_softmax_output_b = self.model.add_parameters((len(self.label_list)))
        self.aux2_softmax_output_w = self.model.add_parameters((len(self.label_list), self.LAYER_SIZE))
        self.aux2_softmax_output_b = self.model.add_parameters((len(self.label_list)))
        self.softmax_output_w = self.model.add_parameters((len(self.label_list), self.LAYER_SIZE * 2))
        self.softmax_output_b = self.model.add_parameters((len(self.label_list)))

    def _make_input(self, seqs):
        chars = []
        tags = []

        for seq in seqs:
            for entry in seq:
                for char_idx in range(len(entry.word)):
                    chars.append(entry.word[char_idx])
                    if len(entry.word) == 1:
                        tags.append('S')
                    elif char_idx == 0:
                        tags.append('B')
                    elif char_idx == len(entry.word) - 1:
                        tags.append('E')
                    else:
                        tags.append('I')
                    if entry.is_compound_entry:
                        tags[-1] = tags[-1] + 'M'

                if "spaceafter=no" not in entry.space_after.lower():
                    chars.append(' ')
            delta = -1
            while tags[delta].startswith('X'):
                delta -= 1  # it should never crash if dataset is ok

            if tags[delta].startswith('S'):
                tags[delta] = 'T'
            else:
                append_m = ''
                if tags[delta].endswith('M'):
                    append_m = 'M'
                tags[delta] = 'U' + append_m
        return chars, tags

    def learn(self, conll_sequences, lang_id=0):
        dy.renew_cg()
        chars, tags = self._make_input(conll_sequences)

        outputs, aux1, aux2 = self._forward(chars, lang_id=lang_id)
        # crf_output = self.crf.tag(outputs)
        tgt_tags = [self.label2int[tag] for tag in tags]

        # loss = self.crf.learn(outputs, tgt_tags) / len(tgt_tags)
        # from ipdb import set_trace
        # set_trace()

        loss_main = dy.esum([-dy.log(dy.pick(output, tgt)) for output, tgt in zip(outputs, tgt_tags)]) / len(tgt_tags)
        loss_aux1 = dy.esum([-dy.log(dy.pick(output, tgt)) for output, tgt in zip(aux1, tgt_tags)]) / len(tgt_tags)
        loss_aux2 = dy.esum([-dy.log(dy.pick(output, tgt)) for output, tgt in zip(aux2, tgt_tags)]) / len(tgt_tags)

        loss = loss_main + 0.5 * loss_aux1 + 0.5 * loss_aux2
        l_val = loss.value() * len(tgt_tags)
        loss.backward()
        self.trainer.update()
        return l_val

    def _batch_forward(self, chars, lang_id, batch_size=1000):
        tags = []
        num_batches = len(chars) // batch_size
        if len(chars) % batch_size != 0:
            num_batches += 1

        for ii in range(num_batches):
            start = ii * batch_size
            stop = ii * batch_size + batch_size
            if stop > len(chars):
                stop = len(chars)

            cbs = stop - start
            offset = 0
            if start > 0:
                start -= self.WINDOW_SIZE * self.NUM_LAYERS
                offset = self.WINDOW_SIZE * self.NUM_LAYERS
            stop = min(len(chars), stop + self.WINDOW_SIZE * self.NUM_LAYERS)

            copy_chars = chars[start:stop]
            dy.renew_cg()
            # from ipdb import set_trace
            # set_trace()
            outputs, _, _ = self._forward(copy_chars, lang_id=lang_id, runtime=True)
            outputs = outputs[offset:offset + cbs]

            new_tags = [np.argmax(pred.npvalue()) for pred in outputs]
            # new_tags = self.crf.tag(outputs)
            for tag in new_tags:
                tags.append(self.label_list[tag])
        return tags

    def tokenize(self, raw_text, lang_id=0):
        # make sequences of approx 2000-4000 chars
        BATCH_SIZE = 1000  # len(raw_text)
        start = 0
        seqs = []
        seq = []
        word_index = 1
        word = ''

        chars = [c for c in raw_text]
        tags = self._batch_forward(chars, batch_size=BATCH_SIZE, lang_id=lang_id)

        for index, char, tag in zip(range(len(tags)), chars, tags):
            if not tag.startswith('X'):
                word += char
            if not tag.startswith('B') and not tag.startswith('X') and not tag.startswith(
                    'I') and word.strip() != '':
                entry = ConllEntry(word_index, word, '_', '_', '_', '_', word_index - 1, '_', '_', '_')
                seq.append(entry)
                word_index += 1
                word = ''
            if tag.startswith('T') or tag.startswith('U'):
                seqs.append(seq)
                seq = []
                word_index = 1
        start += len(chars)

        if word.strip() != '':
            entry = ConllEntry(word_index, word, '_', '_', '_', '_', word_index - 1, '_', '_', '_')
            seq.append(entry)
        if len(seq) != 0:
            seqs.append(seq)

        return seqs

    def _is_known_word(self, chars, index):
        max_win_size = 10
        for zz in range(min(max_win_size, index + 1)):
            word = ''.join(chars[index - zz:index + 1])
            if word in self.dict:
                return dy.scalarInput(1)
        return dy.scalarInput(0)

    def _forward(self, chars, lang_id=0, runtime=True):
        inp_size = self.config.char_emb_size + 32 + self.config.lang_emb_size
        if self.dict is not None:
            inp_size += 1
        inp = [dy.inputVector(np.zeros(inp_size)) for ii in
               range(self.WINDOW_SIZE)]

        lang_emb = self.lang_lookup[lang_id]

        for index, char in zip(range(len(chars)), chars):
            case_emb = self.case_lookup[0]
            if char.lower() == char.upper():
                case_emb = self.case_lookup[1]
            elif char.lower() != char:
                case_emb = self.case_lookup[2]
            if char.lower() in self.encodings.char2int:
                char_emb = self.char_lookup[self.encodings.char2int[char.lower()]]

            else:
                char_emb = self.char_lookup[self.encodings.char2int['<UNK>']]

            if self.dict is None:
                inp.append(dy.concatenate([case_emb, char_emb, lang_emb]))
            else:
                inp.append(dy.concatenate([case_emb, char_emb, lang_emb, self._is_known_word(chars, index)]))

        for ii in range(self.WINDOW_SIZE + 1):
            inp.append(dy.inputVector(np.zeros((inp_size))))

        # outputs = []

        input = [self._proj_w.expr(update=True) * x + self._proj_b.expr(update=True) for x in inp]
        inp_fw = input

        skip_conn_fw = [[] for ii in range(len(chars))]
        for idx, g_w, g_b, a_w, a_b, skip_w, skip_b in zip(range(len(self._fw_gate_w)), self._fw_gate_w,
                                                           self._fw_gate_b,
                                                           self._fw_act_w, self._fw_act_b, self._fw_skip_w,
                                                           self._fw_skip_b):
            new_inp = [dy.inputVector(np.zeros(self.LAYER_SIZE)) for ii in
                       range(self.WINDOW_SIZE)]

            for ii in range(len(chars)):
                hidden = dy.concatenate(inp_fw[ii:ii + self.WINDOW_SIZE + 1])
                skip_conn_fw[ii].append(skip_w.expr(update=True) * hidden + skip_b.expr(update=True))

                act = dy.tanh(a_w.expr(update=True) * hidden + a_b.expr(update=True))
                gate = dy.logistic(g_w.expr(update=True) * hidden + g_b.expr(update=True))
                output = dy.cmult(act, gate) + dy.cmult(1.0 - gate, inp_fw[ii + self.WINDOW_SIZE])

                if not runtime:
                    output = dy.dropout(output, 0.25)
                new_inp.append(output)

            for ii in range(self.WINDOW_SIZE + 1):
                new_inp.append(dy.inputVector(np.zeros((self.LAYER_SIZE))))
            inp_fw = new_inp

        inp_bw = input

        skip_conn_bw = [[] for ii in range(len(chars))]
        for idx, g_w, g_b, a_w, a_b, skip_w, skip_b in zip(range(len(self._bw_gate_w)), self._bw_gate_w,
                                                           self._bw_gate_b,
                                                           self._bw_act_w, self._bw_act_b, self._bw_skip_w,
                                                           self._bw_skip_b):
            new_inp = [dy.inputVector(np.zeros(self.LAYER_SIZE)) for ii in
                       range(self.WINDOW_SIZE)]

            for ii in range(len(chars)):
                hidden = dy.concatenate(inp_bw[ii + self.WINDOW_SIZE: ii + self.WINDOW_SIZE * 2 + 1])
                skip_conn_bw[ii].append(skip_w.expr(update=True) * hidden + skip_b.expr(update=True))

                act = dy.tanh(a_w.expr(update=True) * hidden + a_b.expr(update=True))
                gate = dy.logistic(g_w.expr(update=True) * hidden + g_b.expr(update=True))
                output = dy.cmult(act, gate) + dy.cmult(1.0 - gate, inp_bw[ii + self.WINDOW_SIZE])

                if not runtime:
                    output = dy.dropout(output, 0.25)
                new_inp.append(output)

            for ii in range(self.WINDOW_SIZE + 1):
                new_inp.append(dy.inputVector(np.zeros((self.LAYER_SIZE))))
            inp_bw = new_inp

        outputs_aux1 = [
            dy.softmax(
                self.aux1_softmax_output_w.expr(update=True) * dy.rectify(
                    dy.esum(res)) + self.aux1_softmax_output_b.expr(
                    update=True)) for hidden, res in zip(inp_fw[self.WINDOW_SIZE:], skip_conn_fw)]
        outputs_aux2 = [
            dy.softmax(
                self.aux2_softmax_output_w.expr(update=True) * dy.rectify(
                    dy.esum(res)) + self.aux2_softmax_output_b.expr(
                    update=True)) for hidden, res in zip(inp_bw[self.WINDOW_SIZE:], skip_conn_bw)]

        outputs = [
            dy.softmax(
                self.softmax_output_w.expr(update=True) * dy.rectify(
                    dy.concatenate([dy.esum(resfw), dy.esum(resbw)])) + self.softmax_output_b.expr(
                    update=True)) for hidden, resfw, resbw in
                    zip(inp_fw[self.WINDOW_SIZE:], skip_conn_fw, skip_conn_bw)]
        # outputs = inp[self.window_size:]
        return outputs, outputs_aux1, outputs_aux2

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.populate(filename)


class CRFTokenizer:
    def __init__(self, config, encodings, num_languages=1, runtime=False):
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.encodings = encodings
        self.config = config
        self.char_lookup = self.model.add_lookup_parameters((len(self.encodings.char2int), config.char_emb_size))
        self.case_lookup = self.model.add_lookup_parameters((3, 32))
        input_size = 32 + config.char_emb_size

        self.lang_lookup = self.model.add_lookup_parameters((num_languages, config.lang_emb_size))
        input_size += config.lang_emb_size

        label_list = ['B', 'I', 'E', 'S', 'X', 'BM', 'IM', 'EM', 'SM', 'T', 'U', 'UM']
        self.label_list = []
        self.label2int = {}
        for upos in self.encodings.upos2int:
            for label in label_list:
                key = label + '_' + upos
                self.label2int[key] = len(self.label2int)
                self.label_list.append(key)

        # self.label2int = {'B': 0, 'I': 1, 'E': 2, 'S': 3, 'X': 4, 'BM': 5, 'IM': 6, 'EM': 7, 'SM': 8, 'T': 9, 'U': 10,
        #                   'UM': 11}
        # self.label_list = ['B', 'I', 'E', 'S', 'X', 'BM', 'IM', 'EM', 'SM', 'T', 'U', 'UM']

        self.lstm_fw = []
        self.lstm_bw = []

        for layer_size in self.config.lstm_layers:
            self.lstm_fw.append(dy.LSTMBuilder(1, input_size, layer_size, self.model))
            self.lstm_bw.append(dy.LSTMBuilder(1, input_size, layer_size, self.model))
            input_size = layer_size * 2 + self.config.lang_emb_size

        self.crf_decoder = CRFDecoder(self.model, input_size, 100, len(self.label2int))

    def _make_input(self, seqs):
        chars = []
        tags = []

        for seq in seqs:
            for entry in seq:
                upos = ''
                for char_idx in range(len(entry.word)):
                    chars.append(entry.word[char_idx])
                    upos = entry.upos
                    if len(entry.word) == 1:
                        tags.append('S')
                    elif char_idx == 0:
                        tags.append('B')
                    elif char_idx == len(entry.word) - 1:
                        tags.append('E')
                    else:
                        tags.append('I')
                    if entry.is_compound_entry:
                        tags[-1] = tags[-1] + 'M'

                    tags[-1] = tags[-1] + '_' + upos
                if "spaceafter=no" not in entry.space_after.lower():
                    chars.append(' ')
                    tags.append('X_' + upos)
            delta = -1
            while tags[delta].startswith('X'):
                delta -= 1  # it should never crash if dataset is ok

            if tags[delta].startswith('S'):
                tags[delta] = 'T_' + upos
            else:
                append_m = ''
                if tags[delta][1] == 'M':
                    append_m = 'M'
                tags[delta] = 'U' + append_m + '_' + upos
        return chars, tags

    def learn(self, conll_sequences, lang_id=0):
        dy.renew_cg()
        chars, tags = self._make_input(conll_sequences)
        # for char, tag in zip(chars, tags):
        #     print(char + "-" + tag)
        #
        # for seq in conll_sequences:
        #     for entry in seq:
        #         print(entry.word + "\t" + entry.space_after)
        # sys.exit(0)
        embs = self._forward(chars, lang_id=lang_id)
        tgt_tags = [self.label2int[tag] for tag in tags]
        loss = self.crf_decoder.learn(embs, tgt_tags)
        l_val = loss.value()
        loss.backward()
        self.trainer.update()
        return l_val

    def tokenize(self, raw_text, lang_id=0):
        # make sequences of approx 2000-4000 chars
        BATCH_SIZE = 2000
        start = 0
        seqs = []
        seq = []
        word_index = 1
        word = ''

        while start < len(raw_text):
            dy.renew_cg()
            stop = start + BATCH_SIZE
            if len(raw_text) - stop < BATCH_SIZE:
                stop = len(raw_text)

            chars = [c for c in raw_text[start:stop]]
            embs = self._forward(chars, lang_id=lang_id)
            tags = self.crf_decoder.tag(embs)

            tmp_tags = [self.label_list[tag] for tag in tags]
            tags = tmp_tags

            for index, char, tag in zip(range(len(tags)), chars, tags):
                if not tag.startswith('X'):
                    word += char
                if not tag.startswith('B') and not tag.startswith('X') and not tag.startswith(
                        'I') and word.strip() != '':
                    entry = ConllEntry(word_index, word, '_', '_', '_', '_', word_index - 1, '_', '_', '_')
                    seq.append(entry)
                    word_index += 1
                    word = ''
                if tag.startswith('T') or tag.startswith('U'):
                    seqs.append(seq)
                    seq = []
                    word_index = 1
            start += len(chars)

        if word.strip() != '':
            entry = ConllEntry(word_index, word, '_', '_', '_', '_', word_index - 1, '_', '_', '_')
            seq.append(entry)
        if len(seq) != 0:
            seqs.append(seq)

        return seqs

    def _forward(self, chars, lang_id=0):
        inp = []

        lang_emb = self.lang_lookup[lang_id]

        for char in chars:
            case_emb = self.case_lookup[0]
            if char.lower() == char.upper():
                case_emb = self.case_lookup[1]
            elif char.lower() != char:
                case_emb = self.case_lookup[2]
            if char.lower() in self.encodings.char2int:
                char_emb = self.char_lookup[self.encodings.char2int[char.lower()]]

            else:
                char_emb = self.char_lookup[self.encodings.char2int['<UNK>']]

            inp.append(dy.concatenate([case_emb, char_emb, lang_emb]))

        for l_fw, l_bw in zip(self.lstm_fw, self.lstm_bw):
            x_fw = l_fw.initial_state().transduce(inp)
            x_bw = l_fw.initial_state().transduce(inp)
            inp = [dy.concatenate([fw, bw, lang_emb]) for fw, bw in zip(x_fw, list(reversed(x_bw)))]

        return inp

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.populate(filename)


class TieredTokenizer:
    def __init__(self, config, encodings, embeddings, runtime=False):
        self.config = config
        self.word_embeddings = embeddings
        self.encodings = encodings

        self.modelSS = dy.Model()
        self.modelTok = dy.Model()
        self.trainerSS = dy.AdamTrainer(self.modelSS, alpha=2e-3, beta_1=0.9, beta_2=0.9)
        self.trainerTok = dy.AdamTrainer(self.modelTok, alpha=2e-3, beta_1=0.9, beta_2=0.9)

        # sentence split model
        from cube.generic_networks.wrappers import CNN, CNNConvLayer, CNNPoolingLayer
        from cube.generic_networks.utils import orthonormal_VanillaLSTMBuilder
        # character-level-embeddings
        self.SS_char_lookup = self.modelSS.add_lookup_parameters(
            (len(self.encodings.char2int), self.config.ss_char_embeddings_size))
        self.SS_char_lookup_casing = self.modelSS.add_lookup_parameters((3, 5))  # lower, upper N/A
        self.SS_char_lookup_special = self.modelSS.add_lookup_parameters((2, self.config.ss_char_embeddings_size + 5))
        # lstm-peek network
        if runtime:
            self.SS_peek_lstm = dy.VanillaLSTMBuilder(self.config.ss_peek_lstm_layers,
                                                      self.config.ss_char_embeddings_size + 5,
                                                      self.config.ss_peek_lstm_size, self.modelSS)
        else:
            self.SS_peek_lstm = orthonormal_VanillaLSTMBuilder(self.config.ss_peek_lstm_layers,
                                                               self.config.ss_char_embeddings_size + 5,
                                                               self.config.ss_peek_lstm_size, self.modelSS)
        layer_is = self.config.ss_peek_lstm_size
        self.SS_aux_softmax_peek_w = self.modelSS.add_parameters((2, layer_is))
        self.SS_aux_softmax_peek_b = self.modelSS.add_parameters((2))
        if runtime:
            self.SS_lstm = dy.VanillaLSTMBuilder(self.config.ss_lstm_layers, self.config.ss_char_embeddings_size + 5,
                                                 self.config.ss_lstm_size,
                                                 self.modelSS)
        else:
            self.SS_lstm = orthonormal_VanillaLSTMBuilder(self.config.ss_lstm_layers,
                                                          self.config.ss_char_embeddings_size + 5,
                                                          self.config.ss_lstm_size,
                                                          self.modelSS)

        self.SS_aux_softmax_prev_w = self.modelSS.add_parameters((2, self.config.ss_lstm_size))
        self.SS_aux_softmax_prev_b = self.modelSS.add_parameters((2))

        # post MLP and softmax
        self.SS_mlp_w = []
        self.SS_mlp_b = []
        layer_is = self.config.ss_lstm_size + self.config.ss_peek_lstm_size
        for layer in self.config.ss_mlp_layers:
            self.SS_mlp_w.append(self.modelSS.add_parameters((layer, layer_is)))
            self.SS_mlp_b.append(self.modelSS.add_parameters((layer)))
            layer_is = layer

        self.SS_mlp_softmax_w = self.modelSS.add_parameters((2, layer_is))
        self.SS_mlp_softmax_b = self.modelSS.add_parameters((2))

        # tokenization model
        self.TOK_char_lookup = self.modelTok.add_lookup_parameters(
            (len(self.encodings.char2int), self.config.tok_char_embeddings_size))
        self.TOK_char_lookup_casing = self.modelTok.add_lookup_parameters((3, 5))  # lower, upper N/A
        self.TOK_char_lookup_special = self.modelTok.add_lookup_parameters(
            (2, self.config.tok_char_embeddings_size + 5))
        self.TOK_word_lookup = self.modelTok.add_lookup_parameters(
            (len(self.encodings.word2int), self.config.tok_word_embeddings_size))

        self.TOK_word_embeddings_special = self.modelTok.add_lookup_parameters(
            (2, self.word_embeddings.word_embeddings_size))

        self.TOK_word_proj_w = self.modelTok.add_parameters(
            (self.config.tok_word_embeddings_size, self.word_embeddings.word_embeddings_size))
        self.TOK_word_peek_proj_w = self.modelTok.add_parameters(
            (self.config.tok_word_embeddings_size, self.word_embeddings.word_embeddings_size))
        # lstm networks
        if runtime:
            self.TOK_backward_lstm = dy.VanillaLSTMBuilder(self.config.tok_char_peek_lstm_layers,
                                                           self.config.tok_char_embeddings_size + 5,
                                                           self.config.tok_char_peek_lstm_size, self.modelTok)
            self.TOK_forward_lstm = dy.VanillaLSTMBuilder(self.config.tok_char_lstm_layers,
                                                          self.config.tok_char_embeddings_size + 5,
                                                          self.config.tok_char_lstm_size, self.modelTok)
            self.TOK_word_lstm = dy.VanillaLSTMBuilder(self.config.tok_word_lstm_layers,
                                                       self.config.tok_word_embeddings_size,
                                                       self.config.tok_word_lstm_size,
                                                       self.modelTok)
        else:
            self.TOK_backward_lstm = orthonormal_VanillaLSTMBuilder(self.config.tok_char_peek_lstm_layers,
                                                                    self.config.tok_char_embeddings_size + 5,
                                                                    self.config.tok_char_peek_lstm_size, self.modelTok)
            self.TOK_forward_lstm = orthonormal_VanillaLSTMBuilder(self.config.tok_char_lstm_layers,
                                                                   self.config.tok_char_embeddings_size + 5,
                                                                   self.config.tok_char_lstm_size, self.modelTok)
            self.TOK_word_lstm = orthonormal_VanillaLSTMBuilder(self.config.tok_word_lstm_layers,
                                                                self.config.tok_word_embeddings_size,
                                                                self.config.tok_word_lstm_size,
                                                                self.modelTok)

        self.TOK_mlp_w = []
        self.TOK_mlp_b = []
        layer_input = self.config.tok_word_lstm_size + self.config.tok_char_lstm_size + self.config.tok_char_peek_lstm_size + 2 + self.config.tok_word_embeddings_size
        for layer_size in self.config.tok_mlp_layers:
            self.TOK_mlp_w.append(self.modelTok.add_parameters((layer_size, layer_input)))
            self.TOK_mlp_b.append(self.modelTok.add_parameters((layer_size)))
            layer_input = layer_size

        self.TOK_softmax_w = self.modelTok.add_parameters((2, layer_input))
        self.TOK_softmax_b = self.modelTok.add_parameters((2))
        self.TOK_softmax_peek_w = self.modelTok.add_parameters((2, self.config.tok_char_peek_lstm_size))
        self.TOK_softmax_peek_b = self.modelTok.add_parameters((2))
        self.TOK_softmax_prev_w = self.modelTok.add_parameters((2, self.config.tok_char_lstm_size))
        self.TOK_softmax_prev_b = self.modelTok.add_parameters((2))

        self.losses = []
        self.losses_tok = []

    def start_batch(self):
        self.losses = []
        self.losses_tok = []
        dy.renew_cg()

    def end_batch(self):
        total_loss_val = 0
        if len(self.losses) > 0:
            total_loss_ss = dy.esum(self.losses)
            total_loss_val += total_loss_ss.value()
            total_loss_ss.backward()
            self.trainerSS.update()

        if len(self.losses_tok):
            total_loss_tok = dy.esum(self.losses_tok)
            total_loss_val += total_loss_tok.value()
            total_loss_tok.backward()
            self.trainerTok.update()
        return total_loss_val

    def _predict_tok(self, seq, y_gold=None, runtime=False):
        x_list = []
        offset = 1

        word_is_known = dy.inputVector([1.0, 0.0])
        word_is_unknown = dy.inputVector([0.0, 1.0])

        for char in seq:
            lookup_char = char.lower()
            import re
            lookup_char = re.sub('\d', '0', lookup_char)
            if lookup_char in self.encodings.char2int:
                char_emb = self.TOK_char_lookup[self.encodings.char2int[lookup_char]]
            else:
                char_emb = self.TOK_char_lookup[self.encodings.char2int['<UNK>']]

            if char.lower() == char and char.upper() == char:
                casing_emb = self.TOK_char_lookup_casing[0]  # does not support casing
            elif char.lower() == char:
                casing_emb = self.TOK_char_lookup_casing[1]  # is lowercased
            else:
                casing_emb = self.TOK_char_lookup_casing[2]  # is uppercased

            x = dy.concatenate([char_emb, casing_emb])
            x_list.append(x)

        for _ in range(offset):
            x_list.append(self.TOK_char_lookup_special[1])

        aux_softmax_output_peek = []
        aux_softmax_output_prev = []
        softmax_output = []

        if runtime:
            self.TOK_forward_lstm.set_dropouts(0, 0)
            self.TOK_backward_lstm.set_dropouts(0, 0)
            self.TOK_word_lstm.set_dropouts(0, 0)
        else:
            self.TOK_forward_lstm.set_dropouts(self.config.tok_char_lstm_dropout, self.config.tok_char_lstm_dropout)
            self.TOK_backward_lstm.set_dropouts(self.config.tok_char_peek_lstm_dropout,
                                                self.config.tok_char_peek_lstm_dropout)
            self.TOK_word_lstm.set_dropouts(self.config.tok_word_lstm_dropout, self.config.tok_word_lstm_dropout)

        fw_out = self.TOK_forward_lstm.initial_state().transduce(x_list)
        bw_out = list(reversed(self.TOK_backward_lstm.initial_state().transduce(reversed(x_list))))
        word_lstm = self.TOK_word_lstm.initial_state().add_input(
            dy.inputVector([0] * self.config.tok_word_embeddings_size))
        word = ""
        for index in range(len(seq)):
            word += seq[index]
            aux_softmax_output_prev.append(
                dy.softmax(self.TOK_softmax_prev_w.expr(update=True) * fw_out[index] + self.TOK_softmax_prev_b.expr(
                    update=True)))
            aux_softmax_output_peek.append(
                dy.softmax(self.TOK_softmax_peek_w.expr(update=True) * bw_out[index] + self.TOK_softmax_peek_b.expr(
                    update=True)))

            word_state = word_is_unknown
            peek_emb, found = self.word_embeddings.get_word_embeddings(word.strip())
            if found:
                word_state = word_is_known
                peek_emb = self.TOK_word_peek_proj_w.expr(update=True) * dy.inputVector(peek_emb)
            else:
                peek_emb = self.TOK_word_peek_proj_w.expr(update=True) * self.TOK_word_embeddings_special[0]

            if word.strip().lower() in self.encodings.word2int:
                word_state = word_is_known
                peek_hol = self.TOK_word_lookup[self.encodings.word2int[word.strip().lower()]]
            else:
                peek_hol = self.TOK_word_lookup[self.encodings.word2int['<UNK>']]

            hidden = dy.concatenate(
                [fw_out[index], bw_out[index], word_lstm.output(), word_state, dy.tanh(peek_hol + peek_emb)])
            for w, b, dropout in zip(self.TOK_mlp_w, self.TOK_mlp_b, self.config.tok_mlp_dropouts):
                hidden = dy.tanh(w.expr(update=True) * hidden + b.expr(update=True))
                if not runtime:
                    hidden = dy.dropout(hidden, dropout)

            softmax_output.append(
                dy.softmax(self.TOK_softmax_w.expr(update=True) * hidden + self.TOK_softmax_b.expr(update=True)))
            must_split = False
            if not runtime:
                if y_gold[index] == "S" or y_gold[index] == "SX":
                    must_split = True
            elif np.argmax(softmax_output[-1].npvalue()) == 1:
                must_split = True

            if must_split:
                emb, found = self.word_embeddings.get_word_embeddings(word.strip())
                if not found:
                    emb = self.TOK_word_embeddings_special[1]
                else:
                    emb = dy.inputVector(emb)

                word = word.lower().strip()
                if word in self.encodings.word2int:
                    hol = self.TOK_word_lookup[self.encodings.word2int[word]]
                else:
                    hol = self.TOK_word_lookup[self.encodings.word2int['<UNK>']]

                emb = self.TOK_word_proj_w.expr(update=True) * emb

                word_lstm = word_lstm.add_input(dy.tanh(emb + hol))
                word = ""

        return softmax_output, aux_softmax_output_prev, aux_softmax_output_peek

    def learn_tok(self, ix, iy, aux_softmax_weight=0.2):
        losses = []
        # we must split into sentences before learning tokenization - otherwise, the data will be inconsistent with the runtime version
        start = 0
        import copy
        x = copy.deepcopy(ix)
        y = copy.deepcopy(iy)
        while True:
            if "SX" in y:
                index_of_ss = y.index("SX")
            else:
                break

            y_prediction, y_prediction_aux1, y_prediction_aux2 = self._predict_tok(x[start:index_of_ss + 1],
                                                                                   y_gold=y[start:index_of_ss + 1],
                                                                                   runtime=False)
            for y_real, y_pred, y_aux1, y_aux2 in zip(y, y_prediction, y_prediction_aux1, y_prediction_aux2):
                if y_real == "SX" or y_real == "S":
                    losses.append(-dy.log(dy.pick(y_pred, 1)))
                    losses.append(-dy.log(dy.pick(y_aux1, 1)) * aux_softmax_weight)
                    losses.append(-dy.log(dy.pick(y_aux2, 1)) * aux_softmax_weight)
                else:
                    losses.append(-dy.log(dy.pick(y_pred, 0)))
                    losses.append(-dy.log(dy.pick(y_aux1, 0)) * aux_softmax_weight)
                    losses.append(-dy.log(dy.pick(y_aux2, 0)) * aux_softmax_weight)
            x = x[index_of_ss + 1:]
            y = y[index_of_ss + 1:]

        loss = dy.esum(losses)
        self.losses_tok.append(loss)

    def learn_ss(self, x, y, aux_softmax_weight=0.2):
        losses = []
        y_prediction, y_prediction_aux1, y_prediction_aux2 = self._predict_ss(x, runtime=False)
        for y_real, y_pred, y_aux1, y_aux2 in zip(y, y_prediction, y_prediction_aux1, y_prediction_aux2):
            if y_real == "SX":
                losses.append(-dy.log(dy.pick(y_pred, 1)))
                losses.append(-dy.log(dy.pick(y_aux1, 1)) * aux_softmax_weight)
                losses.append(-dy.log(dy.pick(y_aux2, 1)) * aux_softmax_weight)
            else:
                losses.append(-dy.log(dy.pick(y_pred, 0)))
                losses.append(-dy.log(dy.pick(y_aux1, 0)) * aux_softmax_weight)
                losses.append(-dy.log(dy.pick(y_aux2, 0)) * aux_softmax_weight)

        loss = dy.esum(losses)
        self.losses.append(loss)

    def _get_tokens(self, input_string, space_after_end_of_sentence=True):
        # print("\n")
        # print(input_string)
        tokens = []
        y_pred, _, _ = self._predict_tok(input_string, runtime=True)
        index = 0
        w = ""

        for i in range(len(input_string)):
            w += input_string[i]
            if np.argmax(y_pred[i].npvalue()) == 1:
                if w.strip() != "":
                    index += 1
                    space_after = "SpaceAfter=No"
                    if i < len(input_string) - 1:
                        if input_string[i + 1] in string.whitespace:
                            space_after = "_"
                    entry = ConllEntry(index, str(w).strip(), '_', "_", "_", "_", 0, "_", "_",
                                       space_after=space_after)
                    tokens.append(entry)
                    w = ""

        if w.strip() != "":
            index += 1
            entry = ConllEntry(index, str(w).strip(), '_', "_", "_", "_", 0, "_", "_", "")
            tokens.append(entry)

        # set SpaceAfter=No property of last token
        if len(tokens) > 0:
            tokens[-1].space_after = "SpaceAfter=No" if space_after_end_of_sentence == False else "_"

        return tokens

    def tokenize(self, input_string):
        batch_size = 1000
        import sys
        if sys.version_info[0] == 2:
            input_string = unicode(input_string, 'utf-8')
        else:
            import copy
            input_string = copy.deepcopy(input_string)

        sequences = []
        num_chars = 0
        last_proc = 0
        sz = len(input_string)

        w = ""
        while len(input_string) > 0:
            current_string = input_string[:min(len(input_string), batch_size + 1)]
            """
            proc = num_chars * 100 / sz
            while last_proc + 5 < proc:
                last_proc += 5
                sys.stdout.write(" " + str(last_proc))
                sys.stdout.flush()
            """

            dy.renew_cg()
            y_pred, _, _ = self._predict_ss(current_string)

            last_ss_break = -1
            last_checked_index = -1
            if len(current_string) == batch_size:
                current_string = current_string[:-100]

            for y, char, index in zip(y_pred, current_string, range(len(current_string))):
                w += char
                if np.argmax(y.npvalue()) == 1:
                    space_after_end_of_sentence = False
                    if index < len(
                            input_string) - 1:  # compare with input_string not with current_string for whitespace after current sentence
                        if input_string[index + 1] in string.whitespace:
                            space_after_end_of_sentence = True
                    seq = self._get_tokens(w.strip(), space_after_end_of_sentence=space_after_end_of_sentence)
                    sequences.append(seq)
                    w = ""
                    last_ss_break = index
                last_checked_index = index

            if last_ss_break == -1:  # no sentence break applied
                last_ss_break = last_checked_index
            else:
                w = ""

            input_string = input_string[last_ss_break + 1:]

            num_chars += last_ss_break

        if w.strip() != "":
            space_after_end_of_sentence = False
            if w[-1] in string.whitespace:
                space_after_end_of_sentence = True
            seq = self._get_tokens(w.strip(), space_after_end_of_sentence=space_after_end_of_sentence)
            sequences.append(seq)

        return sequences

    def save_ss(self, filename):
        self.modelSS.save(filename)

    def save_tok(self, filename):
        self.modelTok.save(filename)

    def load(self, filename):
        self.modelTok.populate(filename + "-tok.bestAcc")
        self.modelSS.populate(filename + "-ss.bestAcc")

    def _predict_ss(self, seq, runtime=True):
        x_list = []
        offset = self.config.ss_char_peek_count

        for char in seq:
            lookup_char = char.lower()
            import re
            lookup_char = re.sub('\d', '0', lookup_char)
            if lookup_char in self.encodings.char2int:
                char_emb = self.SS_char_lookup[self.encodings.char2int[lookup_char]]
            else:
                char_emb = self.SS_char_lookup[self.encodings.char2int['<UNK>']]

            if char.lower() == char and char.upper() == char:
                casing_emb = self.SS_char_lookup_casing[0]  # does not support casing
            elif char.lower() == char:
                casing_emb = self.SS_char_lookup_casing[1]  # is lowercased
            else:
                casing_emb = self.SS_char_lookup_casing[2]  # is uppercased

            x = dy.concatenate([char_emb, casing_emb])
            x_list.append(x)

        for _ in range(offset):
            x_list.append(self.SS_char_lookup_special[1])

        aux_softmax_output_peek = []
        aux_softmax_output_prev = []
        softmax_output = []

        # forward pass should not be a problem
        if runtime:
            self.SS_lstm.set_dropouts(0, 0)
            self.SS_peek_lstm.set_dropouts(0, 0)
        else:
            self.SS_lstm.set_dropouts(self.config.ss_lstm_dropout, self.config.ss_lstm_dropout)
            self.SS_peek_lstm.set_dropouts(self.config.ss_peek_lstm_dropout, self.config.ss_peek_lstm_dropout)
        lstm_fw = self.SS_lstm.initial_state()

        for cIndex in range(len(seq)):
            peek_chars = x_list[cIndex:cIndex + self.config.ss_char_peek_count + 1]
            peek_out = self.SS_peek_lstm.initial_state().transduce(reversed(peek_chars))[-1]

            aux_softmax_output_peek.append(
                dy.softmax(self.SS_aux_softmax_peek_w.expr(update=True) * peek_out + self.SS_aux_softmax_peek_b.expr(
                    update=True)))

            lstm_fw = lstm_fw.add_input(x_list[cIndex])
            lstm_out = lstm_fw.output()
            aux_softmax_output_prev.append(
                dy.softmax(self.SS_aux_softmax_prev_w.expr(update=True) * lstm_out + self.SS_aux_softmax_prev_b.expr(
                    update=True)))
            hidden = dy.concatenate([lstm_out, peek_out])
            for w, b, dropout in zip(self.SS_mlp_w, self.SS_mlp_b, self.config.ss_mlp_dropouts):
                hidden = dy.tanh(w.expr(update=True) * hidden + b.expr(update=True))
                if not runtime:
                    hidden = dy.dropout(hidden, dropout)
            softmax_output.append(
                dy.softmax(self.SS_mlp_softmax_w.expr(update=True) * hidden + self.SS_mlp_softmax_b.expr(update=True)))

        return softmax_output, aux_softmax_output_peek, aux_softmax_output_prev


class BDRNNTokenizer:

    def __init__(self, config, encodings, embeddings, runtime=False):
        # INTERNAL PARAMS ###################################################        
        self.config = config
        self.encodings = encodings
        self.word_embeddings = embeddings
        self.config.char_vocabulary_size = len(encodings.characters)
        self.decoder_output_class_count = 3  # O S SX
        self.decoder_output_i2c = {}
        self.decoder_output_i2c[0] = "O"
        self.decoder_output_i2c[1] = "S"
        self.decoder_output_i2c[2] = "SX"
        self.decoder_output_c2i = {}
        self.decoder_output_c2i["O"] = 0
        self.decoder_output_c2i["S"] = 1
        self.decoder_output_c2i["SX"] = 2

        # NETWORK ###########################################################    
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.trainer.set_sparse_updates(False)

        # EMBEDDING SPECIAL TOKENS
        self.word_embeddings_special = self.model.add_lookup_parameters(
            (2, self.word_embeddings.word_embeddings_size))  # [0] = UNK, [1] = SENTENCE START

        # ENCODER-CHAR        
        self.char_embeddings = self.model.add_lookup_parameters(
            (self.config.char_vocabulary_size, self.config.char_embedding_size))
        # self.next_chars_embedding = self.model.add_lookup_parameters(
        #    (self.config.char_vocabulary_size, self.config.next_chars_embedding_size))

        self.char_embeddings_punctuation = self.model.add_lookup_parameters(
            (self.config.char_generic_feature_vocabulary_size, self.config.char_generic_feature_embedding_size))
        self.char_embeddings_whitespace = self.model.add_lookup_parameters(
            (self.config.char_generic_feature_vocabulary_size, self.config.char_generic_feature_embedding_size))
        self.char_embeddings_uppercase = self.model.add_lookup_parameters(
            (self.config.char_generic_feature_vocabulary_size, self.config.char_generic_feature_embedding_size))
        self.encoder_char_input_size = self.config.char_embedding_size + 3 * self.config.char_generic_feature_embedding_size
        if runtime:
            self.encoder_char_lstm1_fw_builder = dy.VanillaLSTMBuilder(1, self.encoder_char_input_size,
                                                                       self.config.encoder_char_lstm_size, self.model)

            self.encoder_char_lstm2_bw_builder = dy.VanillaLSTMBuilder(1,
                                                                       self.config.next_chars_embedding_size + 3 * self.config.char_generic_feature_embedding_size,
                                                                       self.config.encoder_char_lstm_size, self.model)
            self.encoder_word_lstm_builder = dy.VanillaLSTMBuilder(1, self.word_embeddings.word_embeddings_size,
                                                                   self.config.encoder_word_lstm_size, self.model)
        else:
            from generic_networks.utils import orthonormal_VanillaLSTMBuilder
            self.encoder_char_lstm1_fw_builder = orthonormal_VanillaLSTMBuilder(1, self.encoder_char_input_size,
                                                                                self.config.encoder_char_lstm_size,
                                                                                self.model)

            self.encoder_char_lstm2_bw_builder = orthonormal_VanillaLSTMBuilder(1,
                                                                                self.config.next_chars_embedding_size + 3 * self.config.char_generic_feature_embedding_size,
                                                                                self.config.encoder_char_lstm_size,
                                                                                self.model)
            self.encoder_word_lstm_builder = orthonormal_VanillaLSTMBuilder(1,
                                                                            self.word_embeddings.word_embeddings_size,
                                                                            self.config.encoder_word_lstm_size,
                                                                            self.model)

        # ENCODER-WORD

        # self.att_w1 = self.model.add_parameters((
        #     self.config.next_chars_embedding_size + self.config.char_generic_feature_embedding_size * 3,
        #     self.config.encoder_char_lstm_size))
        # self.att_w2 = self.model.add_parameters((
        #     self.config.next_chars_embedding_size + self.config.char_generic_feature_embedding_size * 3,
        #     self.config.encoder_char_lstm_size))
        # self.att_v = self.model.add_parameters(
        #     (1, self.config.next_chars_embedding_size + self.config.char_generic_feature_embedding_size * 3))

        # DECODER

        self.holisticWE = self.model.add_lookup_parameters(
            (len(encodings.word2int), self.word_embeddings.word_embeddings_size))

        self.decoder_input_size = 2 * self.config.encoder_char_lstm_size + self.config.encoder_word_lstm_size + self.word_embeddings.word_embeddings_size

        self.decoder_hiddenW = self.model.add_parameters((self.config.decoder_hidden_size, self.decoder_input_size))
        self.decoder_hiddenB = self.model.add_parameters((self.config.decoder_hidden_size))
        self.decoder_outputW = self.model.add_parameters(
            (self.decoder_output_class_count, self.config.decoder_hidden_size))
        self.decoder_outputB = self.model.add_parameters((self.decoder_output_class_count))

        self.aux_softmax_char_peek_w = self.model.add_parameters(
            (self.decoder_output_class_count, self.config.encoder_char_lstm_size))
        self.aux_softmax_char_peek_b = self.model.add_parameters((self.decoder_output_class_count))

        self.aux_softmax_char_hist_w = self.model.add_parameters(
            (self.decoder_output_class_count, self.config.encoder_char_lstm_size))
        self.aux_softmax_char_hist_b = self.model.add_parameters((self.decoder_output_class_count))
        print("done")

    def learn(self, X, y):  # X is a list of symbols, y is a list of labels for each symbol
        softmax, softmax_aux1, softmax_aux2 = self._predict(X, y)

        losses = []

        for i in range(len(softmax)):
            loss = -dy.log(dy.pick(softmax[i], self.decoder_output_c2i[y[i]]))
            losses.append(loss)
            loss_aux = -dy.log(dy.pick(softmax_aux1[i], self.decoder_output_c2i[y[i]])) * 0.2
            losses.append(loss_aux)
            loss_aux = -dy.log(dy.pick(softmax_aux2[i], self.decoder_output_c2i[y[i]])) * 0.2
            losses.append(loss_aux)
        l = dy.esum(losses)
        loss = l.value()
        l.backward()
        self.trainer.update()
        return loss

    def tokenize(self,
                 input_string):  # input string is a single string that can contain several sentences, output will be conllu-format list of sentences
        import sys
        if sys.version_info[0] == 2:
            uni_string = unicode(input_string, 'utf-8')  # because reasons
        else:
            uni_string = input_string
        offset = 0
        sentences = []
        last_proc = 0
        while offset < len(uni_string):
            proc = (offset + 1) * 100 / len(uni_string)

            while last_proc + 5 < proc:
                last_proc += 5
                sys.stdout.write(" " + str(last_proc))
                sys.stdout.flush()

            # print("Total len = "+str(len(uni_string)))
            # print("Current offset = "+str(offset))
            window = 0
            while True:  # extend window until we find an end of sentence (SX)
                window += self.config.tokenize_maximum_sequence_length
                X = uni_string[offset:min(len(uni_string), offset + window)]
                # print("    X len = "+str(len(X)))
                softmax, _, _ = self._predict(X)
                # print("    Softmax len = "+str(len(softmax)))
                # convert to labels
                labels = [self.decoder_output_i2c[np.argmax(s.npvalue())] for s in softmax]
                # print("    Predicted label len = "+str(len(labels)))
                if "SX" in labels:
                    break
                elif offset + len(labels) >= len(
                        uni_string):  # maybe we reached end of input_string without an SX, then exit as well
                    break
            offset += len(labels)

            # create sentence from labels
            sentence = []
            word = ""
            cnt = 1
            # with fopen("log.txt","a") as log:
            # log.write("\n\n")
            for i in range(len(labels)):
                # log.write("["+X[i].encode('utf-8')+"] "+labels[i]+" w=["+word.encode('utf-8')+"]\n")
                if "O" in labels[i]:
                    word = word + X[i]
                if "S" in labels[i]:
                    if X[i] in string.whitespace:  # if whitespace, skip
                        if word != "":
                            entry = ConllEntry(index=cnt, word=word.decode('utf-8'), lemma="_", upos="_", xpos="_",
                                               attrs="_", head="0",
                                               label="_", deps="_", space_after="_")
                            # log.write("   New ERROR incomplete entry ["+word.encode('utf-8')+"]\n")
                            sentence.append(entry)
                            cnt += 1
                            word = ""
                        continue
                    word += X[i]
                    space_after = "SpaceAfter=No"
                    if i < len(X) - 1:
                        if X[i + 1] in string.whitespace:
                            space_after = "_"
                    entry = ConllEntry(index=cnt, word=word.decode('utf-8'), lemma="_", upos="_", xpos="_", attrs="_",
                                       head="0",
                                       label="_", deps="_", space_after=space_after)
                    # log.write("   New entry ["+word.encode('utf-8')+"]\n")
                    sentence.append(entry)
                    cnt += 1
                    word = ""
            sentences.append(sentence)

            # for entry in sentence:
            #    print(" \t Word :"+entry.word)

        return sentences

    def _attend(self, input_vectors, state):
        w1 = self.att_w1.expr(update=True)
        w2 = self.att_w2.expr(update=True)
        v = self.att_v.expr(update=True)
        attention_weights = []

        w2dt = w2 * state.h()[-1]
        for input_vector in input_vectors:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)

        attention_weights = dy.softmax(dy.concatenate(attention_weights))

        output_vectors = dy.esum(
            [vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])

        return output_vectors

    def _predict(self, X, y=None):  # X is a list of symbols, y is a list of labels for each symbol
        dy.renew_cg()
        runtime = True if y is None else False
        debug = False
        if debug: print("Predict sequence start, len = " + str(len(X)))

        # ENCODER-CHAR
        lstm1_forward = self.encoder_char_lstm1_fw_builder.initial_state()

        # ENCODER-WORD         
        encoder_word_lstm = self.encoder_word_lstm_builder.initial_state()
        encoder_word_lstm = encoder_word_lstm.add_input(self.word_embeddings_special[1])  # [1] = SENTENCE START
        encoder_word_output = encoder_word_lstm.output()

        # PARTIAL-EMBEDDING
        partial_word = self.word_embeddings_special[1]  # [1] = SENTENCE START

        # DECODER
        prediction_output = []
        softmax_dy = []
        softmax_aux_peek = []
        softmax_aux_hist = []
        index_last_word_start = 0

        zeroVecArray_next_chars_embedding = dy.inputVector(
            [0] * (self.config.next_chars_embedding_size + self.config.char_generic_feature_embedding_size * 3))

        for char_index in range(len(X)):  # for each character in X
            if debug: print(" Char index " + str(char_index) + " of " + str(len(X)) + " = \"" + X[char_index] + "\"")

            # ENCODER-CHAR
            encoder_char_input = self._encoder_char_encode_input(X[char_index:char_index + 1],
                                                                 runtime)  # encode only the current symbol
            # print(encoder_char_input)
            lstm1_forward = lstm1_forward.add_input(encoder_char_input[0])
            encoder_char_output = lstm1_forward.output()

            # PARTIAL-EMBEDDING
            if index_last_word_start == 0:
                partial_word = self.word_embeddings_special[1]  # [1] = sentence_start
            else:
                word = ""
                for i in range(index_last_word_start, char_index + 1):
                    word += X[i]
                    # word = word.decode('utf-8')
                partial_word = self._encoder_word_encode_input(word)
                if debug: print("   partial word : [" + word + "]")

            # # NEXT-CHARS MLP
            next_chars = self._next_chars_encode_input(
                X[char_index:min(len(X), char_index + self.config.next_chars_window_size)], runtime)
            remaining = self.config.next_chars_window_size - (len(X) - char_index)
            # # if debug: print("      *NEXT-CHARS: remaining: "+str(remaining)+" current len : "+str(len(next_chars)))
            for r in range(remaining):
                next_chars.append(zeroVecArray_next_chars_embedding)

            next_chars = self.encoder_char_lstm2_bw_builder.initial_state().transduce(reversed(next_chars))
            next_chars = next_chars[-1]  # self._attend(next_chars, lstm1_forward)

            softmax_aux_peek.append(
                dy.softmax(
                    self.aux_softmax_char_peek_w.expr(update=True) * next_chars + self.aux_softmax_char_peek_b.expr(
                        update=True)))
            softmax_aux_hist.append(dy.softmax(
                self.aux_softmax_char_hist_w.expr(
                    update=True) * encoder_char_output + self.aux_softmax_char_hist_b.expr(update=True)))

            # dropout at feature-set level:
            # if runtime:
            decoder_input = dy.concatenate([encoder_char_output, encoder_word_output, partial_word,
                                            next_chars])  # embeddings concatenation of char and word
            # decoder_input = dy.concatenate([encoder_char_output, next_chars])
            if not runtime:
                decoder_input = dy.dropout(decoder_input, self.config.dropout_rate)

            decoder_hidden = dy.tanh(
                self.decoder_hiddenW.expr(update=True) * decoder_input + self.decoder_hiddenB.expr(update=True))
            if not runtime:
                decoder_hidden = dy.dropout(decoder_hidden, self.config.dropout_rate)

            softmax_output = dy.softmax(
                self.decoder_outputW.expr(update=True) * decoder_hidden + self.decoder_outputB.expr(update=True))
            softmax_dy.append(softmax_output)
            if runtime:
                prediction = self.decoder_output_i2c[np.argmax(softmax_output.npvalue())]
            else:  # learn
                prediction = y[
                    char_index]  # so force prediction to be = gold-standard, so we have correct word splits

            if debug: print("    Predicted output: " + prediction)

            if "S" in prediction:
                # compose word
                word = ""
                for i in range(index_last_word_start, char_index + 1):
                    word += X[i]

                index_last_word_start = char_index + 1
                # get ENCODER-WORD output
                current_word_embedding = self._encoder_word_encode_input(word)
                encoder_word_lstm = encoder_word_lstm.add_input(current_word_embedding)
                encoder_word_output = encoder_word_lstm.output()
                # RESET the CHAR LSTM
                lstm1_forward = self.encoder_char_lstm1_fw_builder.initial_state()

            if "X" in prediction:  # stop everything
                break

        if runtime and debug:
            raw_input("STEP")
        return softmax_dy, softmax_aux_peek, softmax_aux_hist

    def _encoder_word_encode_input(self, word):
        embedding, _ = self.word_embeddings.get_word_embeddings(word)
        word = word.lower()
        if word in self.encodings.word2int:
            hol_we = self.holisticWE[self.encodings.word2int[word]]
        else:
            hol_we = self.holisticWE[self.encodings.word2int['<UNK>']]
        if embedding is not None:
            return dy.inputVector(embedding) + hol_we
        else:  # UNKNOWN WORD
            return self.word_embeddings_special[0] + hol_we  # UNK

    def _next_chars_encode_input(self, X, runtime=True):
        encoded = []
        for i in range(len(X)):
            char = X[i]
            char_lower = X[i].lower()
            import re
            char_lower = re.sub('\d', '0', char_lower)
            char = re.sub('\d', '0', char)

            if char_lower in self.encodings.char2int:
                embedding = self.char_embeddings[self.encodings.char2int[char_lower]]
            else:
                embedding = self.char_embeddings[
                    self.encodings.char2int["<UNK>"]]  # dy.inputVector([0]*self.config.next_chars_embedding_size)

            all_info = [embedding]
            if char_lower != char:
                all_info.append(self.char_embeddings_uppercase[1])
            else:
                all_info.append(self.char_embeddings_uppercase[0])

            if char_lower in string.whitespace:
                all_info.append(self.char_embeddings_whitespace[1])
            else:
                all_info.append(self.char_embeddings_whitespace[0])

            # is punctuation
            if char_lower in string.punctuation:
                all_info.append(self.char_embeddings_punctuation[1])
            else:
                all_info.append(self.char_embeddings_punctuation[0])

            encoded.append(dy.concatenate(all_info))
        return encoded

    def _encoder_char_encode_input(self, X, runtime=True):
        encoded = []
        for char_index in range(len(X)):
            embeddings = []
            # char itself
            char = X[char_index]
            char_lower = char.lower()
            import re
            char_lower = re.sub('\d', '0', char_lower)
            char = re.sub('\d', '0', char)
            if char_lower not in self.encodings.char2int:
                # print (" UNK CHAR: ["+char+"]")
                char_lower = "<UNK>"
            embeddings.append(self.char_embeddings[self.encodings.char2int[char_lower]])
            # is uppercase
            if char.lower() != char:
                embeddings.append(self.char_embeddings_uppercase[1])
            else:
                embeddings.append(self.char_embeddings_uppercase[0])
            # is whitespace
            if char in string.whitespace:
                embeddings.append(self.char_embeddings_whitespace[1])
            else:
                embeddings.append(self.char_embeddings_whitespace[0])
            # is punctuation
            if char in string.punctuation:
                embeddings.append(self.char_embeddings_punctuation[1])
            else:
                embeddings.append(self.char_embeddings_punctuation[0])

            input = dy.concatenate(embeddings)
            # if not runtime:
            #    input = dy.noise(input,self.config.encoder_char_input_noise)
            encoded.append(input)
        return encoded

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.populate(path)

    def _argmax(self, probs):
        max_index = 0
        for zz in range(len(probs)):
            if probs[zz] > probs[max_index]:
                max_index = zz
        return max_index
