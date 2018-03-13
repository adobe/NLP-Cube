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
from character_embeddings import CharacterNetwork


class BDRNNLemmatizer:
    def __init__(self, lemmatizer_config, encodings, embeddings, runtime=False):
        self.config = lemmatizer_config
        self.encodings = encodings
        self.embeddings = embeddings
        self.losses=[]

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9, beta_2=0.9)

        self.character_network = CharacterNetwork(self.config.tag_embeddings_size, encodings,
                                                  rnn_size=self.config.char_rnn_size,
                                                  rnn_layers=self.config.char_rnn_layers,
                                                  embeddings_size=self.config.char_embeddings,
                                                  model=self.model, runtime=runtime)

        self.upos_lookup = self.model.add_lookup_parameters(
            (len(self.encodings.upos2int), self.config.tag_embeddings_size))
        self.xpos_lookup = self.model.add_lookup_parameters(
            (len(self.encodings.xpos2int), self.config.tag_embeddings_size))
        self.attrs_lookup = self.model.add_lookup_parameters(
            (len(self.encodings.attrs2int), self.config.tag_embeddings_size))
        self.char_lookup=self.model.add_lookup_parameters((len(self.encodings.char2int),self.config.char_embeddings))

        if runtime:
            self.rnn = dy.LSTMBuilder(self.config.rnn_layers,
                                  self.config.char_rnn_size * 2 + self.config.char_embeddings, self.config.rnn_size,
                                  self.model)
        else:
            from utils import orthonormal_VanillaLSTMBuilder
            self.rnn = orthonormal_VanillaLSTMBuilder(self.config.rnn_layers,
                                      self.config.char_rnn_size * 2 + self.config.char_embeddings, self.config.rnn_size,
                                      self.model)

        self.att_w1 = self.model.add_parameters((200, self.config.char_rnn_size * 2))
        self.att_w2 = self.model.add_parameters((200, self.config.rnn_size+self.config.tag_embeddings_size))
        self.att_v = self.model.add_parameters((1, 200))

        self.start_lookup = self.model.add_lookup_parameters(
            (1, self.config.char_rnn_size * 2 + self.config.char_embeddings))

        self.softmax_w = self.model.add_parameters((len(self.encodings.char2int) + 1, self.config.rnn_size))
        self.softmax_b = self.model.add_parameters((len(self.encodings.char2int) + 1))
        self.softmax_casing_w = self.model.add_parameters((2, self.config.rnn_size))
        self.softmax_casing_b = self.model.add_parameters((2))

    def _attend(self, input_vectors, state, embeddings):
        w1 = self.att_w1.expr()
        w2 = self.att_w2.expr()
        v = self.att_v.expr()
        attention_weights = []

        w2dt = w2 * dy.concatenate([state.h()[-1], embeddings])
        for input_vector in input_vectors:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)

        attention_weights = dy.softmax(dy.concatenate(attention_weights))

        output_vectors = dy.esum(
            [vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])

        return output_vectors

    def _predict(self, word, upos, xpos, attrs, num_chars=0, gs_chars=None):
        if num_chars == 0:
            runtime = True
        else:
            runtime = False

        char_emb, states = self.character_network.compute_embeddings(word, runtime=runtime)


        num_predictions = 0
        softmax_list = []
        m1, m2, m3 = 0, 0, 0
        zero_vec = dy.vecInput(self.config.tag_embeddings_size)
        if upos in self.encodings.upos2int:
            upos_emb = self.upos_lookup[self.encodings.upos2int[upos]]
            m1 = 1
        else:
            upos_emb = zero_vec


        if xpos in self.encodings.xpos2int:
            xpos_emb = self.xpos_lookup[self.encodings.xpos2int[xpos]]
            m2 = 1
        else:
            xpos_emb = zero_vec


        if attrs in self.encodings.attrs2int:
            attrs_emb = self.attrs_lookup[self.encodings.attrs2int[attrs]]
            m3 = 1
        else:
            attrs_emb = zero_vec

        scale = float(4.0) / (m1 + m2 + m3 + 1.0)

        scale = dy.scalarInput(scale)
        tag_emb = (upos_emb + xpos_emb + attrs_emb + char_emb) * scale
        rnn = self.rnn.initial_state().add_input(self.start_lookup[0])
        char_emb=dy.inputVector([0]*self.config.char_embeddings)

        while True:
            attention = self._attend(states, rnn, tag_emb)

            input = dy.concatenate([attention, char_emb])
            rnn = rnn.add_input(input)

            softmax = dy.softmax(self.softmax_w.expr() * rnn.output() + self.softmax_b.expr())
            softmax_casing = dy.softmax(self.softmax_casing_w.expr() * rnn.output() + self.softmax_casing_b.expr())
            softmax_list.append([softmax, softmax_casing])
            if num_chars == 0:
                s_index=np.argmax(softmax.npvalue())
                if s_index == len(self.encodings.char2int):
                    break
                char_emb = self.char_lookup[s_index]
            else:
                if num_predictions<len(gs_chars):
                    char=gs_chars[num_predictions]
                    if char in self.encodings.char2int:
                        char_emb = self.char_lookup[self.encodings.char2int[char]]
                    else:
                        char_emb=self.char_lookup[self.encodings.char2int["<UNK>"]]

            num_predictions += 1
            if num_predictions == num_chars or num_predictions > 255:
                break

        return softmax_list

    def start_batch(self):
        self.losses=[]
        dy.renew_cg()

    def end_batch(self):
        total_loss=0
        if len(self.losses)>0:
            loss=dy.esum(self.losses)
            total_loss=loss.value()
            loss.backward()
            self.trainer.update()
        self.losses=[]
        return total_loss

    def learn(self, seq):
        total_loss=0
        for entry in seq:
            if entry.upos != 'NUM' and entry.upos != 'PROPN':
                losses = []
                unilemma = unicode(entry.lemma, 'utf-8')
                n_chars = len(unilemma)
                softmax_output_list = self._predict(entry.word, entry.upos, entry.xpos, entry.attrs, num_chars=n_chars + 1, gs_chars=unilemma)
                #print unilemma.encode('utf-8')#, softmax_output_list
                for softmax, char in zip(softmax_output_list[:-1], unilemma):

                    char_index = -1
                    if char.lower() == char:
                        casing = 0
                    else:
                        casing = 1
                    char = char.lower()
                    if char in self.encodings.char2int:
                        char_index = self.encodings.char2int[char]
                    if char_index != -1:
                        losses.append(-dy.log(dy.pick(softmax[0], char_index)))
                    losses.append(-dy.log(dy.pick(softmax[1], casing)))
                    #print np.argmax(softmax[0].npvalue()), char_index, softmax

                losses.append(-dy.log(dy.pick(softmax_output_list[-1][0], len(self.encodings.char2int))))
                loss = dy.esum(losses)
                self.losses.append(loss)


    def tag(self, seq):
        dy.renew_cg()
        lemmas = []
        for entry in seq:
            if entry.upos=='NUM' or entry.upos == 'PROPN':
                lemma=entry.word.decode('utf-8')
            else:
                softmax_output_list = self._predict(entry.word, entry.upos, entry.xpos, entry.attrs)
                lemma = ""
                for softmax in softmax_output_list[:-1]:
                    char_index=np.argmax(softmax[0].npvalue())
                    if char_index<len(self.encodings.characters):
                        char = self.encodings.characters[char_index]
                        if np.argmax(softmax[1].npvalue()) == 1:
                            char = char.upper()
                        lemma += char
            lemmas.append(lemma)
        return lemmas

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.populate(path)
