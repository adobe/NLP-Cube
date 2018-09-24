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

import numpy as np
import random
import dynet as dy
from cube.generic_networks.character_embeddings import CharacterNetwork
from cube.graph.decoders import GreedyDecoder
from cube.generic_networks.utils import orthonormal_VanillaLSTMBuilder
import copy
import sys


class BDRNNParser:
    def __init__(self, parser_config, encodings, embeddings, aux_softmax_weight=0.2, runtime=False):
        self.config = parser_config
        self.encodings = encodings
        self.embeddings = embeddings
        self.decoder = GreedyDecoder()

        self.model = dy.Model()

        # self.trainer = dy.SimpleSGDTrainer(self.model)
        self.trainer = dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9, beta_2=0.9)

        self.trainer.set_sparse_updates(False)
        self.character_network = CharacterNetwork(100, encodings, rnn_size=200, rnn_layers=1,
                                                  embeddings_size=self.config.input_embeddings_size,
                                                  model=self.model, runtime=runtime)

        self.holistic_embeddings = self.model.add_lookup_parameters(
            (len(self.encodings.word2int), self.config.input_embeddings_size))

        self.input_proj_w_word = self.model.add_parameters(
            (self.config.input_embeddings_size, self.embeddings.word_embeddings_size))
        self.input_proj_b_word = self.model.add_parameters((self.config.input_embeddings_size))

        self.unknown_word_embedding = self.model.add_lookup_parameters(
            (3, self.config.input_embeddings_size))  # for padding lexical
        self.pad_tag_embedding = self.model.add_lookup_parameters(
            (3, self.config.input_embeddings_size))  # for padding morphology

        self.bdrnn_fw = []
        self.bdrnn_bw = []

        rnn_input_size = 0
        if self.config.use_lexical:
            rnn_input_size += self.config.input_embeddings_size

        if self.config.use_morphology:
            rnn_input_size += self.config.input_embeddings_size
            self.upos_lookup = self.model.add_lookup_parameters(
                (len(self.encodings.upos2int), self.config.input_embeddings_size))
            self.xpos_lookup = self.model.add_lookup_parameters(
                (len(self.encodings.xpos2int), self.config.input_embeddings_size))
            self.attrs_lookup = self.model.add_lookup_parameters(
                (len(self.encodings.attrs2int), self.config.input_embeddings_size))

        index = 0
        aux_proj_input_size = 0
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
                aux_proj_input_size = rnn_input_size

        proj_input_size = self.config.layers[-1] * 2

        self.proj_arc_w_head = self.model.add_parameters((self.config.arc_proj_size, proj_input_size))
        self.proj_arc_b_head = self.model.add_parameters((self.config.arc_proj_size))
        self.proj_arc_w_dep = self.model.add_parameters((self.config.arc_proj_size, proj_input_size))
        self.proj_arc_b_dep = self.model.add_parameters((self.config.arc_proj_size))
        self.proj_label_w_head = self.model.add_parameters((self.config.label_proj_size, proj_input_size))
        self.proj_label_b_head = self.model.add_parameters((self.config.label_proj_size))
        self.proj_label_w_dep = self.model.add_parameters((self.config.label_proj_size, proj_input_size))
        self.proj_label_b_dep = self.model.add_parameters((self.config.label_proj_size))
        if not self.config.predict_morphology:
            self.aux_proj_arc_w_head = self.model.add_parameters((self.config.arc_proj_size, aux_proj_input_size))
            self.aux_proj_arc_b_head = self.model.add_parameters((self.config.arc_proj_size))
            self.aux_proj_arc_w_dep = self.model.add_parameters((self.config.arc_proj_size, aux_proj_input_size))
            self.aux_proj_arc_b_dep = self.model.add_parameters((self.config.arc_proj_size))
        else:
            self.upos_proj_w = self.model.add_parameters((self.config.label_proj_size, aux_proj_input_size))
            self.xpos_proj_w = self.model.add_parameters((self.config.label_proj_size, aux_proj_input_size))
            self.attrs_proj_w = self.model.add_parameters((self.config.label_proj_size, aux_proj_input_size))
            self.upos_proj_b = self.model.add_parameters((self.config.label_proj_size))
            self.xpos_proj_b = self.model.add_parameters((self.config.label_proj_size))
            self.attrs_proj_b = self.model.add_parameters((self.config.label_proj_size))

        self.link_b = self.model.add_parameters((1, self.config.arc_proj_size))
        self.link_w = self.model.add_parameters((self.config.arc_proj_size, self.config.arc_proj_size))

        self.label_ww = self.model.add_parameters((1, len(self.encodings.label2int)))
        self.label_w = self.model.add_parameters((len(self.encodings.label2int), self.config.label_proj_size * 2))
        self.label_bb = self.model.add_parameters((len(self.encodings.label2int)))

        if not self.config.predict_morphology:
            self.aux_link_w = self.model.add_parameters((self.config.arc_proj_size, self.config.arc_proj_size))
            self.aux_link_b = self.model.add_parameters((1, self.config.arc_proj_size))
        else:
            self.upos_softmax_w = self.model.add_parameters((len(self.encodings.upos2int), self.config.label_proj_size))
            self.xpos_softmax_w = self.model.add_parameters((len(self.encodings.xpos2int), self.config.label_proj_size))
            self.attrs_softmax_w = self.model.add_parameters(
                (len(self.encodings.attrs2int), self.config.label_proj_size))

            self.upos_softmax_b = self.model.add_parameters((len(self.encodings.upos2int)))
            self.xpos_softmax_b = self.model.add_parameters((len(self.encodings.xpos2int)))
            self.attrs_softmax_b = self.model.add_parameters((len(self.encodings.attrs2int)))
            self.lemma_softmax_b = self.model.add_parameters((len(self.encodings.char2int) + 1))
            self.lemma_softmax_casing_b = self.model.add_parameters((2))

        self.aux_softmax_weight = aux_softmax_weight
        self.batch_loss = []

    def start_batch(self):
        dy.renew_cg()
        self.batch_loss = []

    def end_batch(self):
        if len(self.batch_loss) > 0:
            loss = dy.esum(self.batch_loss)
            loss_val = loss.value()
            loss.backward()
            self.trainer.update()
            return loss_val
        else:
            return 0

    def learn(self, seq):
        # remove compound words
        tmp = []
        for ss in seq:
            if not ss.is_compound_entry:
                tmp.append(ss)
        seq = tmp
        arc_matrix, aux_arc_matrix, proj_labels, softmax_morphology = self._predict_arc(seq, runtime=False)
        gold_heads = [entry.head for entry in seq]
        gold_labels = [entry.label for entry in seq]

        softmax_labels = self._predict_label(gold_heads, proj_labels, runtime=False)

        losses = []

        for gold_head, gold_label, arc_probs, softmax_label, entry in zip(gold_heads, gold_labels,
                                                                          arc_matrix[1:],
                                                                          softmax_labels, seq):
            label_index = self.encodings.label2int[gold_label]
            losses.append(-dy.log(arc_probs[gold_head]))
            losses.append(-dy.log(dy.pick(softmax_label, label_index)))

        if not self.config.predict_morphology:
            for gold_head, aux_probs, entry in zip(gold_heads, aux_arc_matrix[
                                                               1:], seq):
                losses.append(-dy.log(aux_probs[gold_head]) * self.aux_softmax_weight)

        else:
            for softmax_morph, entry in zip(softmax_morphology, seq):
                loss_upos = -dy.log(dy.pick(softmax_morph[0], self.encodings.upos2int[entry.upos]))
                losses.append(loss_upos * (self.aux_softmax_weight / 3))

                if len(
                        self.encodings.xpos2int) > 1:  # stability check (some languages are missing attributes or XPOS, resulting in numerical overflow during backpropagation
                    loss_xpos = -dy.log(dy.pick(softmax_morph[1], self.encodings.xpos2int[entry.xpos]))
                    losses.append(loss_xpos * (self.aux_softmax_weight / 3))

                if len(
                        self.encodings.attrs2int) > 1:  # stability check (some languages are missing attributes or XPOS, resulting in numerical overflow during backpropagation
                    loss_attrs = -dy.log(dy.pick(softmax_morph[2], self.encodings.attrs2int[entry.attrs]))
                    losses.append(loss_attrs * (self.aux_softmax_weight / 3))

        loss = dy.esum(losses)
        self.batch_loss.append(loss)

    def _attend(self, input_vectors, state, aux_embeddings):
        w1 = self.lemma_att_w1.expr(update=True)
        w2 = self.lemma_att_w2.expr(update=True)
        v = self.lemma_att_v.expr(update=True)
        attention_weights = []

        w2dt = w2 * dy.concatenate([state.h()[-1], aux_embeddings])
        for input_vector in input_vectors:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)

        attention_weights = dy.softmax(dy.concatenate(attention_weights))

        output_vectors = dy.esum(
            [vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])

        return output_vectors

    def tag(self, seq):
        tmp = []
        for ss in seq:
            if not ss.is_compound_entry:
                tmp.append(ss)

        # if len(tmp)<2:
        #     print "ERRRORR"
        #     for entry in seq:
        #         print str(entry.index)+"\t"+str(entry.word)
        seq = tmp

        dy.renew_cg()
        arc_matrix, aux_arc_matrix, proj_labels, softmax_morphology = self._predict_arc(seq)
        pred_heads = self.decoder.decode(arc_matrix)
        softmax_labels = self._predict_label(pred_heads, proj_labels)

        tag_list = []
        for pred_head, softmax_label in zip(pred_heads, softmax_labels):
            label_index = np.argmax(softmax_label.npvalue())
            tag = ParserTag(pred_head, self.encodings.labels[label_index], None, None, None)
            tag_list.append(tag)

        if self.config.predict_morphology:
            for tag, softmax_morph in zip(tag_list, softmax_morphology):
                tag.upos = self.encodings.upos_list[np.argmax(softmax_morph[0].npvalue())]
                tag.xpos = self.encodings.xpos_list[np.argmax(softmax_morph[1].npvalue())]
                tag.attrs = self.encodings.attrs_list[np.argmax(softmax_morph[2].npvalue())]

        return tag_list

    def _predict_label(self, heads, proj_labels, runtime=True):
        s_labels = []
        for iDep, iHead in zip(range(1, len(heads) + 1), heads):
            modw = dy.transpose(
                dy.reshape(proj_labels[iHead][1], (self.config.label_proj_size, 1)) * self.label_ww.expr(update=True))
            term1 = modw * proj_labels[iDep][0]
            term2 = self.label_w.expr(update=True) * dy.concatenate([proj_labels[iHead][1], proj_labels[iDep][0]])
            term3 = self.label_bb.expr(update=True)
            s_labels.append(dy.softmax(term1 + term2 + term3))

        return s_labels

    def _make_input(self, seq, runtime):
        x_list = []
        encoder_states_list = [None]
        # add the root
        if not self.config.use_morphology:
            x_list.append(self.unknown_word_embedding[1])
        elif not self.config.use_lexical:
            x_list.append(self.pad_tag_embedding[1])
        else:  # both lexical and morphology are used
            x_list.append(dy.concatenate(
                [self.unknown_word_embedding[1], self.pad_tag_embedding[1]]))

        for entry in seq:
            word = entry.word

            if self.config.use_lexical:
                # prepare lexical embeddings
                char_emb, encoder_states = self.character_network.compute_embeddings(word, runtime=runtime)
                encoder_states_list.append(encoder_states)
                if sys.version_info[0] == 2:
                    word_emb, found = self.embeddings.get_word_embeddings(word.decode('utf-8'))
                else:
                    word_emb, found = self.embeddings.get_word_embeddings(word)
                if not found:
                    word_emb = self.unknown_word_embedding[0]
                else:
                    word_emb = dy.tanh(
                        self.input_proj_w_word.expr(update=True) * dy.inputVector(word_emb) + self.input_proj_b_word.expr(update=True))
                if sys.version_info[0] == 2:
                    word = word.decode('utf-8').lower()
                else:
                    word = word.lower()

                if word in self.encodings.word2int:
                    holistic_emb = self.holistic_embeddings[self.encodings.word2int[word]]
                else:
                    holistic_emb = self.holistic_embeddings[self.encodings.word2int['<UNK>']]

                # dropout lexical embeddings
                if runtime:
                    w_emb = word_emb + char_emb + holistic_emb
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
                    w_emb = (word_emb * m1 + char_emb * m2 + holistic_emb * m3) * scale

            if self.config.use_morphology:
                if entry.upos in self.encodings.upos2int:
                    upos_emb = self.upos_lookup[self.encodings.upos2int[entry.upos]]
                else:
                    upos_emb = dy.inputVector([0] * self.config.input_embeddings_size)
                if entry.xpos in self.encodings.xpos2int:
                    xpos_emb = self.xpos_lookup[self.encodings.xpos2int[entry.xpos]]
                else:
                    xpos_emb = dy.inputVector([0] * self.config.input_embeddings_size)
                if entry.attrs in self.encodings.attrs2int:
                    attrs_emb = self.attrs_lookup[self.encodings.attrs2int[entry.attrs]]
                else:
                    attrs_emb = dy.inputVector([0] * self.config.input_embeddings_size)
                # overwrite all dropouts. it will later be handled by "same-mask"
                t_emb = upos_emb + xpos_emb + attrs_emb
                # w_emb = word_emb + char_emb + holistic_emb

            # compose embeddings, if necessary             
            if self.config.use_lexical and self.config.use_morphology:
                if not runtime:
                    p1 = random.random()
                    p2 = random.random()
                    m1 = 1
                    m2 = 1
                    if p1 < self.config.input_dropout_prob:
                        m1 = 0
                    if p2 < self.config.input_dropout_prob:
                        m2 = 0
                    if m1 + m2 > 0:
                        scale = float(2.0) / (m1 + m2)
                    else:
                        scale = 1.0
                    scale = dy.scalarInput(scale)
                    m1 = dy.scalarInput(m1)
                    m2 = dy.scalarInput(m2)
                    x_list.append(dy.concatenate([w_emb * m1 * scale, t_emb * m2 * scale]))
                else:
                    x_list.append(dy.concatenate([w_emb, t_emb]))
            elif self.config.use_lexical:  # just use_lexical == True
                x_list.append(w_emb)
            else:  # just use_morphology == True
                x_list.append(t_emb)

        # close sequence
        if not self.config.use_morphology:
            x_list.append(self.unknown_word_embedding[2])
        elif not self.config.use_lexical:
            x_list.append(self.pad_tag_embedding[2])
        else:
            x_list.append(
                dy.concatenate(
                    [self.unknown_word_embedding[2], self.pad_tag_embedding[2]]))

        encoder_states_list.append(None)
        return x_list, encoder_states_list

    def _predict_arc(self, seq, runtime=True):
        x_list, encoder_states_list = self._make_input(seq, runtime)

        # BDLSTM
        rnn_outputs = [x_list]
        for fw, bw, dropout in zip(self.bdrnn_fw, self.bdrnn_bw, self.config.layer_dropouts):
            if runtime:
                fw.set_dropouts(0, 0)
                bw.set_dropouts(0, 0)
            else:
                fw.set_dropouts(dropout, dropout)
                bw.set_dropouts(dropout, dropout)

            fw_list = fw.initial_state().transduce(x_list)
            bw_list = list(reversed(bw.initial_state().transduce(reversed(x_list))))
            x_list = [dy.concatenate([x_fw, x_bw]) for x_fw, x_bw in zip(fw_list, bw_list)]

            rnn_outputs.append(x_list)

        # projections
        arc_projections = [[dy.tanh(self.proj_arc_w_dep.expr(update=True) * x + self.proj_arc_b_dep.expr(update=True)),
                            dy.tanh(self.proj_arc_w_head.expr(update=True) * x + self.proj_arc_b_head.expr(update=True))] for x in
                           rnn_outputs[-1]]
        label_projections = [[dy.tanh(self.proj_label_w_dep.expr(update=True) * x + self.proj_label_b_dep.expr(update=True)),
                              dy.tanh(self.proj_label_w_head.expr(update=True) * x + self.proj_label_b_head.expr(update=True))] for x in
                             rnn_outputs[-1]]
        if not runtime:
            arc_projections = [
                [dy.dropout(x1, self.config.presoftmax_mlp_dropout), dy.dropout(x2, self.config.presoftmax_mlp_dropout)]
                for x1, x2 in arc_projections]
            label_projections = [
                [dy.dropout(x1, self.config.presoftmax_mlp_dropout), dy.dropout(x2, self.config.presoftmax_mlp_dropout)]
                for x1, x2 in label_projections]
        if not self.config.predict_morphology:
            aux_arc_projections = [[dy.tanh(self.aux_proj_arc_w_dep.expr(update=True) * x + self.aux_proj_arc_b_dep.expr(update=True)),
                                    dy.tanh(self.aux_proj_arc_w_head.expr(update=True) * x + self.aux_proj_arc_b_head.expr(update=True))]
                                   for x in rnn_outputs[self.config.aux_softmax_layer]]
            if not runtime:
                aux_arc_projections = [[dy.dropout(x1, self.config.presoftmax_mlp_dropout),
                                        dy.dropout(x2, self.config.presoftmax_mlp_dropout)] for x1, x2 in
                                       aux_arc_projections]

        else:
            drp = self.config.presoftmax_mlp_dropout
            if runtime:
                drp = 0
            upos_softmax = [dy.softmax(self.upos_softmax_w.expr(update=True) * dy.dropout(dy.tanh(
                self.upos_proj_w.expr(update=True) * x + self.upos_proj_b.expr(update=True)), drp) + self.upos_softmax_b.expr(update=True)) for x in
                            rnn_outputs[self.config.aux_softmax_layer]]
            xpos_softmax = [dy.softmax(self.xpos_softmax_w.expr(update=True) * dy.dropout(dy.tanh(
                self.xpos_proj_w.expr(update=True) * x + self.xpos_proj_b.expr(update=True)), drp) + self.xpos_softmax_b.expr(update=True)) for x in
                            rnn_outputs[self.config.aux_softmax_layer]]
            attrs_softmax = [dy.softmax(self.attrs_softmax_w.expr(update=True) * dy.dropout(dy.tanh(
                self.attrs_proj_w.expr(update=True) * x + self.attrs_proj_b.expr(update=True)), drp) + self.attrs_softmax_b.expr(update=True)) for x in
                             rnn_outputs[self.config.aux_softmax_layer]]

            morphology_softmax = [[upos, xpos, attrs] for
                                  upos, xpos, attrs in
                                  zip(upos_softmax, xpos_softmax, attrs_softmax)]

        n = len(seq) + 1
        arc_matrix = [[None] * n for _ in range(n)]
        if not self.config.predict_morphology:
            aux_arc_matrix = [[None] * n for _ in range(n)]
        for iDst in range(n):
            term_bias = self.link_b.expr(update=True) * arc_projections[iDst][1]
            term_weight = self.link_w.expr(update=True) * arc_projections[iDst][1]
            if not self.config.predict_morphology:
                aux_term_bias = self.aux_link_b.expr(update=True) * aux_arc_projections[iDst][1]
                aux_term_weight = self.aux_link_w.expr(update=True) * aux_arc_projections[iDst][1]
            for iSrc in range(n):
                if iSrc != iDst:
                    attention = dy.reshape(term_weight, (1, self.config.arc_proj_size)) * arc_projections[iSrc][
                        0] + term_bias
                    arc_matrix[iSrc][iDst] = attention
                    if not self.config.predict_morphology:
                        aux_attention = dy.reshape(aux_term_weight, (1, self.config.arc_proj_size)) * \
                                        aux_arc_projections[iSrc][0] + aux_term_bias
                        aux_arc_matrix[iSrc][iDst] = aux_attention

        # compute softmax for arcs
        a_m = [[None] * n for _ in range(n)]
        if not self.config.predict_morphology:
            aux_a_m = [[None] * n for _ in range(n)]

        for iSrc in range(n):
            s_max = []
            if not self.config.predict_morphology:
                aux_s_max = []
            for iDst in range(n):
                if iSrc != iDst:
                    s_max.append(arc_matrix[iSrc][iDst])
                    if not self.config.predict_morphology:
                        aux_s_max.append(aux_arc_matrix[iSrc][iDst])
            s_max = dy.softmax(dy.concatenate(s_max))
            if not self.config.predict_morphology:
                aux_s_max = dy.softmax(dy.concatenate(aux_s_max))
            ofs = 0
            for iDst in range(n):
                if iSrc == iDst:
                    ofs = -1
                else:
                    a_m[iSrc][iDst] = s_max[iDst + ofs]
                    if not self.config.predict_morphology:
                        aux_a_m[iSrc][iDst] = aux_s_max[iDst + ofs]
        if not self.config.predict_morphology:
            return a_m, aux_a_m, label_projections, None
        else:
            return a_m, None, label_projections, morphology_softmax[1:-1]

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.populate(path)

    def parse_sequences(self, sequences):
        new_sequences = []
        for sequence in sequences:
            new_sequence = copy.deepcopy(sequence)
            predicted_tags = self.tag(new_sequence)
            iOrig, iTags = 0, 0
            while iOrig < len(new_sequence):
                while new_sequence[iOrig].is_compound_entry:
                    iOrig += 1
                new_sequence[iOrig].head = predicted_tags[iTags].head
                new_sequence[iOrig].label = predicted_tags[iTags].label
                if self.config.predict_morphology == True:
                    new_sequence[iOrig].upos = predicted_tags[iTags].upos
                    new_sequence[iOrig].xpos = predicted_tags[iTags].xpos
                    new_sequence[iOrig].attrs = predicted_tags[iTags].attrs
                iTags += 1
                iOrig += 1

            new_sequences.append(new_sequence)
        return new_sequences


class ParserTag:
    def __init__(self, head, label, upos=None, xpos=None, attrs=None, lemma=None):
        self.head = head
        self.label = label
        self.upos = upos
        self.xpos = xpos
        self.attrs = attrs
        self.lemma = lemma
