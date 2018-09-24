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


def get_link(seq, iSrc, iDst):
    l1 = seq[iSrc].label
    l2 = seq[iDst].label

    if iSrc == 0 and l2 != '*':
        return 1
    if iDst == 0 and l1 != '*':
        return 1

    if l1 == "*" or l2 == "*":
        return 0

    pp_l1 = l1.split(';')
    pp_l2 = l2.split(';')
    for l1 in pp_l1:
        for l2 in pp_l2:
            ppl1 = l1.split(':')[0]
            ppl2 = l2.split(':')[0]
            if ppl1 == ppl2:
                return 1
    return 0


def _has_index(index, label):
    parts = label.split(";")
    for part in parts:
        pp = part.split(":")
        if pp[0] == str(index):
            return True
    return False


class GDBNer:
    def __init__(self, config, encodings, embeddings, runtime=False):
        self.model = dy.Model()
        self.config = config
        self.encodings = encodings
        self.embeddings = embeddings

        self.trainer = dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9, beta_2=0.9)

        self.word_lookup = self.model.add_lookup_parameters((len(self.encodings.word2int), self.config.embeddings_size))
        self.upos_lookup = self.model.add_lookup_parameters((len(self.encodings.upos2int), self.config.embeddings_size))
        self.xpos_lookup = self.model.add_lookup_parameters((len(self.encodings.xpos2int), self.config.embeddings_size))
        self.attrs_lookup = self.model.add_lookup_parameters(
            (len(self.encodings.attrs2int), self.config.embeddings_size))

        if self.config.use_char_embeddings:

            from cube.generic_networks.character_embeddings import CharacterNetwork
            self.character_network = CharacterNetwork(self.config.embeddings_size, encodings,
                                                      rnn_size=self.config.char_rnn_size,
                                                      rnn_layers=self.config.char_rnn_layers,
                                                      embeddings_size=self.config.embeddings_size, model=self.model,
                                                      runtime=runtime)

        self.we_proj = self.model.add_parameters(
            (self.config.embeddings_size, self.embeddings.word_embeddings_size))

        self.encoder_fw = []
        self.encoder_bw = []

        lstm_builder = dy.VanillaLSTMBuilder
        if not runtime:
            from cube.generic_networks.utils import orthonormal_VanillaLSTMBuilder
            lstm_builder = orthonormal_VanillaLSTMBuilder

        input_size = self.config.embeddings_size

        for layer_size in self.config.arc_rnn_layers:
            self.encoder_fw.append(lstm_builder(1, input_size, layer_size, self.model))
            self.encoder_bw.append(lstm_builder(1, input_size, layer_size, self.model))
            input_size = layer_size * 2

        self.link_w = self.model.add_parameters((1, self.config.proj_size * 2))
        self.link_b = self.model.add_parameters((1))

        self.label_decoder = lstm_builder(1, self.config.proj_size, self.config.label_rnn_size, self.model)

        self.proj_w1 = self.model.add_parameters((self.config.proj_size, input_size))
        self.proj_w2 = self.model.add_parameters((self.config.proj_size, input_size))
        self.proj_w3 = self.model.add_parameters((self.config.proj_size, input_size))
        self.proj_b1 = self.model.add_parameters((self.config.proj_size))
        self.proj_b2 = self.model.add_parameters((self.config.proj_size))
        self.proj_b3 = self.model.add_parameters((self.config.proj_size))

        self.label_w = self.model.add_parameters((len(self.encodings.label2int), self.config.label_rnn_size))
        self.label_b = self.model.add_parameters((len(self.encodings.label2int)))

        self.losses = []

    def start_batch(self):
        self.losses = []
        dy.renew_cg()

    def end_batch(self):
        total_loss = 0
        if len(self.losses) > 0:
            loss = dy.esum(self.losses)
            total_loss = loss.value()
            loss.backward()
            self.trainer.update()
        return total_loss

    def _make_input(self, seq, runtime=False):
        seq_input = []
        zero_vec_tag = dy.inputVector([0 for i in range(self.config.embeddings_size)])
        for entry in seq:
            word = entry.word
            upos = entry.upos
            xpos = entry.xpos
            attrs = entry.attrs

            word_vec, found = self.embeddings.get_word_embeddings(word)
            if not found:
                word_vec, found = self.embeddings.get_word_embeddings('</s>')

            word_vector = self.we_proj.expr(update=True) * dy.inputVector(word_vec)

            tag_mult = 1.0
            if upos in self.encodings.upos2int:
                upos_vec = self.upos_lookup[self.encodings.upos2int[upos]]
            else:
                upos_vec = zero_vec_tag
                tag_mult += 1.0

            if xpos in self.encodings.xpos2int:
                xpos_vec = self.xpos_lookup[self.encodings.xpos2int[xpos]]
            else:
                xpos_vec = zero_vec_tag
                tag_mult += 1.0

            if attrs is self.encodings.attrs2int:
                attrs_vec = self.attrs_lookup[self.encodings.attrs2int[attrs]]
            else:
                attrs_vec = zero_vec_tag
                tag_mult += 1.0

            tag_vector = (upos_vec + xpos_vec + attrs_vec) * dy.scalarInput(tag_mult)

            if self.config.use_char_embeddings:
                char_vector, states = self.character_network.compute_embeddings(word, runtime=runtime)
            else:
                char_vector = zero_vec_tag

            p1 = np.random.random()
            p2 = np.random.random()
            p3 = np.random.random()
            if not self.config.use_char_embeddings:
                p3 = 1.0

            scale = 1.0
            if not runtime:
                if p1 < 0.34:
                    word_vector = zero_vec_tag
                    scale += 1.0
                if p2 < 0.34:
                    tag_vector = zero_vec_tag
                    scale += 1.0
                if p3 < 0.34:
                    char_vector = zero_vec_tag
                    scale += 1.0
            seq_input.append((word_vector + tag_vector + char_vector) * dy.scalarInput(scale))

        return seq_input

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.populate(path)

    def tag(self, seq):
        dy.renew_cg()
        output, proj_x = self._predict(seq, runtime=True)
        return self._decode(output, proj_x)

    def _predict(self, seq, runtime=True):
        x_list = self._make_input(seq, runtime=runtime)

        for fw, bw in zip(self.encoder_fw, self.encoder_bw):
            x_fw = fw.initial_state().transduce(x_list)
            x_bw = list(reversed(bw.initial_state().transduce(reversed(x_list))))
            x_list = [dy.concatenate([x1, x2]) for x1, x2 in zip(x_fw, x_bw)]

        proj_x1 = [dy.tanh(self.proj_w1.expr(update=True) * x + self.proj_b1.expr(update=True)) for x in x_list]
        proj_x2 = [dy.tanh(self.proj_w2.expr(update=True) * x + self.proj_b2.expr(update=True)) for x in x_list]
        proj_x3 = [dy.tanh(self.proj_w3.expr(update=True) * x + self.proj_b3.expr(update=True)) for x in x_list]

        output = []
        for iSrc in range(len(seq)):
            out_row = []
            for iDst in range(len(seq)):
                if iDst > iSrc:
                    x = dy.concatenate([proj_x1[iSrc], proj_x2[iDst]])
                    out_row.append(dy.logistic(self.link_w.expr(update=True) * x + self.link_b.expr(update=True)))
                else:
                    out_row.append(None)
            output.append(out_row)
        return output, proj_x3

    def _get_gs_chains(self, seq):
        indices = []
        for row in seq:
            if row.label != "*":
                parts = row.label.split(";")
                for part in parts:
                    pp = part.split(":")
                    expr_index = int(pp[0])
                    if expr_index not in indices:
                        indices.append(expr_index)

        chains = []
        labels = []
        for index in indices:
            first = True
            lst = []
            label = ""
            i = 0
            for row in seq:
                if _has_index(index, row.label):
                    if first:
                        first = False
                        parts = row.label.split(";")
                        for part in parts:
                            pp = part.split(":")
                            if len(pp) == 1:
                                print(str(row.index) + "\t" + row.word + "\t" + row.label)
                            if pp[0] == str(index):
                                label = pp[1]
                                break

                    lst.append(i)
                i += 1
            if label == "":
                for row in seq:
                    print (row.orig_line)
            chains.append(lst)
            labels.append(label)

        return chains, labels

    def _valid(self, a, current_nodes, node):
        for other_node in current_nodes:
            if a[node, other_node] == 0:
                return False
        return True

    def _backtrack(self, a, current_nodes, solutions):
        recursed = False
        for i in range(a.shape[0]):
            if a[current_nodes[-1], i] == 1:
                if i not in current_nodes:
                    if self._valid(a, current_nodes, i):
                        current_nodes.append(i)  # push
                        recursed = True
                        self._backtrack(a, current_nodes, solutions)
                        current_nodes = current_nodes[:-1]  # pop
        if not recursed and len(current_nodes) > 1:
            import copy
            solutions.append(copy.deepcopy(current_nodes))

    def learn(self, seq):
        output, proj_x3 = self._predict(seq, runtime=False)

        # arcs
        for iSrc in range(len(seq)):
            for iDst in range(len(seq)):
                if iDst > iSrc:
                    o = output[iSrc][iDst]  # the softmax portion
                    t = get_link(seq, iSrc, iDst)
                    # if t==1:
                    # self.losses.append(-dy.log(dy.pick(o, t)))
                    self.losses.append(dy.binary_log_loss(o, dy.scalarInput(t)))

        # labels
        gs_chains, labels = self._get_gs_chains(seq)

        for chain, label in zip(gs_chains, labels):
            label_rnn = self.label_decoder.initial_state()
            for index in chain:
                label_rnn = label_rnn.add_input(proj_x3[index])
            label_softmax = dy.softmax(
                self.label_w.expr(update=True) * label_rnn.output() + self.label_b.expr(update=True))
            self.losses.append(-dy.log(dy.pick(label_softmax, self.encodings.label2int[label])))

    def _decode(self, output, proj_x):
        expressions = []
        labels = []
        a = np.zeros((len(output), len(output)))
        for iSrc in range(len(output)):
            for iDst in range(len(output)):
                if iDst > iSrc:
                    if output[iSrc][iDst].value() >= 0.5:
                        a[iSrc][iDst] = 1
                        a[iDst][iSrc] = 1

        for iSrc in range(len(output)):
            exprs = []
            current_nodes = [iSrc]
            self._backtrack(a, current_nodes, exprs)
            [expr.sort() for expr in exprs]
            # check for duplicates

            for expr in exprs:
                valid = True
                for e_expr in expressions:
                    if e_expr == expr:
                        valid = False
                        break
                if valid:
                    expressions.append(expr)

        for expression in expressions:
            lstm_label = self.label_decoder.initial_state()
            for index in expression:
                lstm_label = lstm_label.add_input(proj_x[index])
            label_soft = self.label_w.expr(update=True) * lstm_label.output() + self.label_b.expr(update=True)
            label_index = np.argmax(label_soft.npvalue())
            labels.append(self.encodings.labels[label_index])

        return expressions, labels
