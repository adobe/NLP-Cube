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
import random
from cube.generic_networks.utils import orthonormal_VanillaLSTMBuilder


class BRNNMT:
    def __init__(self, src_we, dst_we, input_encodings, output_encodings, config):
        self.config = config
        self.losses = []
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.src_we = src_we
        self.dst_we = dst_we
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings
        # encoder
        self.encoder_fw = []
        self.encoder_bw = []
        input_size = config.input_size
        for layer_size in self.config.encoder_layers:
            self.encoder_fw.append(orthonormal_VanillaLSTMBuilder(1, input_size, layer_size, self.model))
            self.encoder_bw.append(orthonormal_VanillaLSTMBuilder(1, input_size, layer_size, self.model))
            input_size = layer_size * 2

        # decoder
        #self.decoder = []
        #for layer_size in self.config.decoder_layers:
        self.decoder=orthonormal_VanillaLSTMBuilder(config.decoder_layers, input_size+self.config.input_size, config.decoder_size, self.model)
        input_size = config.decoder_size

        # output softmax
        self.output_softmax_w = self.model.add_parameters((len(self.output_encodings.word2int) + 1, input_size))
        self.output_softmax_b = self.model.add_parameters((len(self.output_encodings.word2int) + 1))
        self.EOS = len(self.output_encodings.word2int)
        # aux WE layer
        self.aux_layer_w = self.model.add_parameters(
            (self.config.aux_we_layer_size, self.config.decoder_size))
        self.aux_layer_b = self.model.add_parameters((self.config.aux_we_layer_size))
        # aux WE projection
        self.aux_layer_proj_w = self.model.add_parameters((self.dst_we.word_embeddings_size, self.config.aux_we_layer_size))
        self.aux_layer_proj_b = self.model.add_parameters((self.dst_we.word_embeddings_size))

        # input projection
        self.word_proj_w = self.model.add_parameters((self.config.input_size, self.src_we.word_embeddings_size))
        self.word_proj_b = self.model.add_parameters((self.config.input_size))
        self.hol_we_src = self.model.add_lookup_parameters((len(self.input_encodings.word2int), self.config.input_size))
        self.hol_we_dst = self.model.add_lookup_parameters((len(self.output_encodings.word2int), self.config.input_size))
        self.special_we = self.model.add_lookup_parameters((2, self.config.input_size))

        # attention
        self.att_w1 = self.model.add_parameters(
            (self.config.encoder_layers[-1] * 2, self.config.encoder_layers[-1] * 2))
        self.att_w2 = self.model.add_parameters((self.config.encoder_layers[-1] * 2, self.config.decoder_size))
        self.att_v = self.model.add_parameters((1, self.config.encoder_layers[-1] * 2))

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

    def start_batch(self):
        self.losses = []
        dy.renew_cg()

    def end_batch(self):
        loss_val = 0
        if len(self.losses) > 0:
            loss = dy.esum(self.losses)
            loss_val = loss.value()
            loss.backward()
            self.trainer.update()
        return loss_val

    def translate(self, src):
        dy.renew_cg()
        softmax_list, aux_list = self._predict(src, runtime=True)
        UNK = self.output_encodings.word2int['<UNK>']
        dst = []
        for softmax, aux in zip(softmax_list[:-1], aux_list[:-1]):
            w_index = np.argmax(softmax.npvalue())
            if w_index != UNK:
                dst.append(self.output_encodings.hol_word_list[w_index])
            else:  # this word should be searched inside the target word embeddings based on the value of AUX
                #dst.append(self.dst_we.get_closest_word(aux.value()))
                dst.append('<UNK>')

        return dst

    def learn(self, src, dst):
        softmax_list, aux_list = self._predict(src, dst=dst,  num_predictions=len(dst) + 1, runtime=False)
        for softmax, aux, entry in zip(softmax_list, aux_list, dst):
            word = entry.word.decode('utf-8').lower()
            if word in self.output_encodings.word2int:
                w_index = self.output_encodings.word2int[word]
            else:
                w_index = self.output_encodings.word2int["<UNK>"]

            w_emb, found = self.dst_we.get_word_embeddings(entry.word.decode('utf-8'))
            self.losses.append(-dy.log(dy.pick(softmax, w_index)))
            if found:
                vec1=aux
                vec2=dy.inputVector(w_emb)
                cosine = dy.dot_product(vec1, vec2) * dy.pow(dy.l2_norm(vec1) * dy.l2_norm(vec2),
                                                                       dy.scalarInput(-1))
                self.losses.append(dy.squared_distance(cosine, dy.scalarInput(1.0)))


        self.losses.append(-dy.log(dy.pick(softmax_list[-1], self.EOS)))

    def _make_input(self, list, runtime=True):
        x_list = [self.special_we[0]]
        w_emb_zero = dy.inputVector([0] * self.src_we.word_embeddings_size)

        for entry in list:
            w_emb, found = self.src_we.get_word_embeddings(entry.word.decode('utf-8'))
            if not found:
                w_emb = w_emb_zero
            else:
                w_emb = dy.inputVector(w_emb)
            word = entry.word.decode('utf-8').lower()
            if word in self.input_encodings.word2int:
                hol_emb = self.hol_we_src[self.input_encodings.word2int[word]]
            else:
                hol_emb = self.hol_we_src[self.input_encodings.word2int["<UNK>"]]

            proj_emb = self.word_proj_w.expr(update=True) * w_emb + self.word_proj_b.expr(update=True)
            if runtime:
                x = dy.tanh(proj_emb + hol_emb)
            else:
                p1 = random.random()
                p2 = random.random()
                m1 = 1
                m2 = 1
                if p1 < self.config.input_dropout_prob:
                    m1 = 0
                if p2 < self.config.input_dropout_prob:
                    m2 = 0

                scale = 1.0
                if m1 + m2 > 0:
                    scale = float(2.0) / (m1 + m2)
                m1 = dy.scalarInput(m1)
                m2 = dy.scalarInput(m2)
                scale = dy.scalarInput(scale)
                x = dy.tanh((proj_emb * m1 + hol_emb * m2) * scale)

            x_list.append(x)

        x_list.append(self.special_we[1])
        return x_list

    def _predict(self, src, dst=None, num_predictions=-1, runtime=True):
        # input
        x_list = self._make_input(src, True)
        # encoder
        for fw, bw, dropout in zip(self.encoder_fw, self.encoder_bw, self.config.encoder_layer_dropouts):
            if runtime:
                fw.set_dropouts(0, 0)
                bw.set_dropouts(0, 0)
            else:
                fw.set_dropouts(0, dropout)
                bw.set_dropouts(0, dropout)

            fw_list = fw.initial_state().transduce(x_list)
            bw_list = list(reversed(bw.initial_state().transduce(reversed(x_list))))
            x_list = [dy.concatenate([fw_value, bw_value]) for fw_value, bw_value in zip(fw_list, bw_list)]

        # decoder

        predictions_left = num_predictions
        decoder = self.decoder.initial_state().add_input(dy.inputVector([0] * (self.config.encoder_layers[-1] * 2+self.config.input_size)))
        last_dst_we=self.special_we[0]
        softmax_output = []
        aux_output = []
        pred_index=0
        while predictions_left != 0:
            predictions_left -= 1
            input = dy.concatenate([self._attend(x_list, decoder),last_dst_we])

            decoder = decoder.add_input(input)
            softmax = dy.softmax(self.output_softmax_w.expr(update=True) * decoder.output() + self.output_softmax_b.expr(update=True))
            softmax_output.append(softmax)

            proj = dy.tanh(self.aux_layer_w.expr(update=True) * decoder.output() + self.aux_layer_b.expr(update=True))
            aux = self.aux_layer_proj_w.expr(update=True) * proj + self.aux_layer_proj_b.expr(update=True)
            aux_output.append(aux)
            if runtime:
                out_we_index=np.argmax(softmax.npvalue())
                if out_we_index == self.EOS:
                    break
                last_dst_we=self.hol_we_dst[out_we_index]
            else:
                if pred_index<len(dst):
                    last_word=dst[pred_index].word.decode('utf-8').lower()
                    last_word_index=self.output_encodings.word2int['<UNK>']
                    if last_word in self.output_encodings.word2int:
                        last_word_index=self.output_encodings.word2int[last_word]
                    last_dst_we=self.hol_we_dst[last_word_index]
                    pred_index+=1
            #failsafe
            if len(softmax_output) >= 2*len(src):
                break

        return softmax_output, aux_output

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.populate(path)
