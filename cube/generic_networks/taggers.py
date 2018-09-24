#
# Author: Tiberiu Boros
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

import dynet as dy
import numpy as np
import sys
import copy
import random

from cube.generic_networks.character_embeddings import CharacterNetwork
from cube.generic_networks.utils import orthonormal_VanillaLSTMBuilder


class BDRNNTagger:
    def __init__(self, tagger_config, encodings, embeddings, aux_softmax_weight=0.2, runtime=False):
        self.config = tagger_config
        self.encodings = encodings
        self.embeddings = embeddings

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9,
                                      beta_2=0.9)  # dy.MomentumSGDTrainer(self.model)
        self.trainer.set_sparse_updates(False)
        self.character_network = CharacterNetwork(100, encodings, rnn_size=200, rnn_layers=1,
                                                  embeddings_size=self.embeddings.word_embeddings_size,
                                                  model=self.model, runtime=runtime)

        self.unknown_word_embedding = self.model.add_lookup_parameters((1, self.embeddings.word_embeddings_size))
        self.holistic_word_embedding = self.model.add_lookup_parameters(
            (len(encodings.word2int), self.embeddings.word_embeddings_size))

        self.char_proj_w = self.model.add_parameters((self.config.input_size, self.embeddings.word_embeddings_size))
        self.emb_proj_w = self.model.add_parameters((self.config.input_size, self.embeddings.word_embeddings_size))
        self.hol_proj_w = self.model.add_parameters((self.config.input_size, self.embeddings.word_embeddings_size))

        self.bdrnn_fw = []
        self.bdrnn_bw = []
        rnn_input_size = self.config.input_size  # self.embeddings.word_embeddings_size

        aux_softmax_input_size = 0
        index = 0
        for layer_size in self.config.layers:
            if runtime:
                self.bdrnn_fw.append(dy.VanillaLSTMBuilder(1, rnn_input_size, layer_size, self.model))
                self.bdrnn_bw.append(dy.VanillaLSTMBuilder(1, rnn_input_size, layer_size, self.model))
            else:
                self.bdrnn_fw.append(orthonormal_VanillaLSTMBuilder(1, rnn_input_size, layer_size, self.model))
                self.bdrnn_bw.append(orthonormal_VanillaLSTMBuilder(1, rnn_input_size, layer_size, self.model))
            rnn_input_size = layer_size * 2
            index += 1
            if index == self.config.aux_softmax_layer:
                aux_softmax_input_size = rnn_input_size

        self.mlps = []
        for _ in range(3):  # upos, xpos and attrs
            mlp_w = []
            mlp_b = []
            input_sz = self.config.layers[-1] * 2
            for l_size in self.config.presoftmax_mlp_layers:
                mlp_w.append(self.model.add_parameters((l_size, input_sz)))
                mlp_b.append(self.model.add_parameters((l_size)))
                input_sz = l_size
            self.mlps.append([mlp_w, mlp_b])

        softmax_input_size = self.config.presoftmax_mlp_layers[-1]
        self.softmax_upos_w = self.model.add_parameters((len(self.encodings.upos2int), softmax_input_size))
        self.softmax_upos_b = self.model.add_parameters((len(self.encodings.upos2int)))
        self.softmax_xpos_w = self.model.add_parameters((len(self.encodings.xpos2int), softmax_input_size))
        self.softmax_xpos_b = self.model.add_parameters((len(self.encodings.xpos2int)))
        self.softmax_attrs_w = self.model.add_parameters((len(self.encodings.attrs2int), softmax_input_size))
        self.softmax_attrs_b = self.model.add_parameters((len(self.encodings.attrs2int)))

        self.aux_softmax_upos_w = self.model.add_parameters((len(self.encodings.upos2int), aux_softmax_input_size))
        self.aux_softmax_upos_b = self.model.add_parameters((len(self.encodings.upos2int)))
        self.aux_softmax_xpos_w = self.model.add_parameters((len(self.encodings.xpos2int), aux_softmax_input_size))
        self.aux_softmax_xpos_b = self.model.add_parameters((len(self.encodings.xpos2int)))
        self.aux_softmax_attrs_w = self.model.add_parameters((len(self.encodings.attrs2int), aux_softmax_input_size))
        self.aux_softmax_attrs_b = self.model.add_parameters((len(self.encodings.attrs2int)))

        self.aux_softmax_weight = aux_softmax_weight
        self.losses = []

    def tag(self, seq):
        dy.renew_cg()
        softmax_list, aux_softmax_list = self._predict(seq)
        label_list = []
        for softmax in softmax_list:
            label_list.append([self.encodings.upos_list[np.argmax(softmax[0].npvalue())],
                               self.encodings.xpos_list[np.argmax(softmax[1].npvalue())],
                               self.encodings.attrs_list[np.argmax(softmax[2].npvalue())]])
        return label_list

    def learn(self, seq):
        # dy.renew_cg()
        softmax_list, aux_softmax_list = self._predict(seq, runtime=False)
        losses = []
        for entry, softmax, aux_softmax in zip(seq, softmax_list, aux_softmax_list):
            upos_index = self.encodings.upos2int[entry.upos]
            xpos_index = self.encodings.xpos2int[entry.xpos]
            attrs_index = self.encodings.attrs2int[entry.attrs]

            losses.append(-dy.log(dy.pick(softmax[0], upos_index)))
            losses.append(-dy.log(dy.pick(softmax[1], xpos_index)))
            losses.append(-dy.log(dy.pick(softmax[2], attrs_index)))
            losses.append(-dy.log(dy.pick(aux_softmax[0], upos_index)) * (self.aux_softmax_weight / 3))
            losses.append(-dy.log(dy.pick(aux_softmax[1], xpos_index)) * (self.aux_softmax_weight / 3))
            losses.append(-dy.log(dy.pick(aux_softmax[2], attrs_index)) * (self.aux_softmax_weight / 3))

        # loss = dy.average(losses)
        # loss_val = loss.value()
        # loss.backward()
        # self.trainer.update()
        # return loss_val
        self.losses.append(dy.esum(losses))

    def start_batch(self):
        self.losses = []
        dy.renew_cg()

    def end_batch(self):
        total_loss_val = 0
        if len(self.losses) > 0:
            total_loss = dy.esum(self.losses)
            self.losses = []
            total_loss_val = total_loss.value()
            total_loss.backward()
            self.trainer.update()
        return total_loss_val

    def _predict(self, seq, runtime=True):
        softmax_list = []
        aux_softmax_list = []
        x_list = []
        for entry in seq:
            word = entry.word
            char_emb, _ = self.character_network.compute_embeddings(word, runtime=runtime)
            import sys
            if sys.version_info[0] == 2:
                word_emb, found = self.embeddings.get_word_embeddings(word.decode('utf-8'))
            else:
                word_emb, found = self.embeddings.get_word_embeddings(word)
            if not found:
                word_emb = self.unknown_word_embedding[0]
            else:
                word_emb = dy.inputVector(word_emb)
            if sys.version_info[0] == 2:
                holistic_word = word.decode('utf-8').lower()
            else:
                holistic_word = word.lower()
            if holistic_word in self.encodings.word2int:
                hol_emb = self.holistic_word_embedding[self.encodings.word2int[holistic_word]]
            else:
                hol_emb = self.holistic_word_embedding[self.encodings.word2int['<UNK>']]
            proj_emb = self.emb_proj_w.expr(update=True) * word_emb
            proj_hol = self.hol_proj_w.expr(update=True) * hol_emb
            proj_char = self.char_proj_w.expr(update=True) * char_emb
            # x_list.append(dy.tanh(proj_char + proj_emb + proj_hol))

            if runtime:
                x_list.append(dy.tanh(proj_char + proj_emb + proj_hol))
            else:
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                m1 = 1
                m2 = 1
                m3 = 1
                if p1 < self.config.input_dropout_prob:
                    m1 = 0
                if p2 < self.config.input_dropout_prob:
                    m2 = 0
                if p3 < self.config.input_dropout_prob:
                    m3 = 0

                scale = 1.0
                if m1 + m2 + m3 > 0:
                    scale = float(3) / (m1 + m2 + m3)
                m1 = dy.scalarInput(m1)
                m2 = dy.scalarInput(m2)
                m3 = dy.scalarInput(m3)
                scale = dy.scalarInput(scale)
                x_list.append(dy.tanh((proj_char * m1 + proj_emb * m2 + proj_hol * m3) * scale))

        # BDLSTM
        rnn_outputs = []
        for fw, bw, dropout in zip(self.bdrnn_fw, self.bdrnn_bw, self.config.layer_dropouts):
            if not runtime:
                fw.set_dropouts(0, dropout)
                bw.set_dropouts(0, dropout)
            else:
                fw.set_dropouts(0, 0)
                bw.set_dropouts(0, 0)
            fw_list = fw.initial_state().transduce(x_list)
            bw_list = list(reversed(bw.initial_state().transduce(reversed(x_list))))
            x_list = [dy.concatenate([x_fw, x_bw]) for x_fw, x_bw in zip(fw_list, bw_list)]
            # if runtime:
            #    x_out = x_list
            # else:
            #    x_out = [dy.dropout(x, dropout) for x in x_list]
            rnn_outputs.append(x_list)

        # SOFTMAX
        mlp_output = []
        for x in rnn_outputs[-1]:
            pre_softmax = []
            for iMLP in range(3):
                mlp_w = self.mlps[iMLP][0]
                mlp_b = self.mlps[iMLP][1]
                inp = x
                for w, b, drop, in zip(mlp_w, mlp_b, self.config.presoftmax_mlp_dropouts):
                    inp = dy.tanh(w.expr(update=True) * inp + b.expr(update=True))
                    if not runtime:
                        inp = dy.dropout(inp, drop)
                pre_softmax.append(inp)
            mlp_output.append(pre_softmax)

        for softmax_inp, aux_softmax_inp in zip(mlp_output, rnn_outputs[self.config.aux_softmax_layer - 1]):
            softmax_list.append([dy.softmax(self.softmax_upos_w.expr(update=True) * softmax_inp[0] + self.softmax_upos_b.expr(update=True)),
                                 dy.softmax(self.softmax_xpos_w.expr(update=True) * softmax_inp[1] + self.softmax_xpos_b.expr(update=True)),
                                 dy.softmax(
                                     self.softmax_attrs_w.expr(update=True) * softmax_inp[2] + self.softmax_attrs_b.expr(update=True))])
            aux_softmax_list.append(
                [dy.softmax(self.aux_softmax_upos_w.expr(update=True) * aux_softmax_inp + self.aux_softmax_upos_b.expr(update=True)),
                 dy.softmax(self.aux_softmax_xpos_w.expr(update=True) * aux_softmax_inp + self.aux_softmax_xpos_b.expr(update=True)),
                 dy.softmax(self.aux_softmax_attrs_w.expr(update=True) * aux_softmax_inp + self.aux_softmax_attrs_b.expr(update=True))])

        return softmax_list, aux_softmax_list

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.populate(path)

    def tag_sequences(self, sequences):
        new_sequences = []
        for sequence in sequences:
            new_sequence = copy.deepcopy(sequence)
            predicted_tags = self.tag(new_sequence)
            for entryIndex, pred in enumerate(predicted_tags):
                new_sequence[entryIndex].upos = pred[0]
                new_sequence[entryIndex].xpos = pred[1]
                new_sequence[entryIndex].attrs = pred[2]
            new_sequences.append(new_sequence)
        return new_sequences
