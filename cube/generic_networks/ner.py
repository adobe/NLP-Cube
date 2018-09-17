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

        self.we_proj = self.model.add_lookup_parameters(
            (self.config.embeddings_size, self.embeddings.word_embeddings_size))

        self.encoder_fw = []
        self.encoder_bw = []

        lstm_builder = dy.VanillaLSTMBuilder
        if not runtime:
            from utils import orthonormal_VanillaLSTMBuilder
            lstm_builder = orthonormal_VanillaLSTMBuilder

        input_size = self.config.embeddings_size

        for layer_size in self.config.arc_rnn_size:
            self.encoder_fw.append(lstm_builder(1, input_size, layer_size, self.model))
            self.encoder_bw.append(lstm_builder(1, input_size, layer_size, self.model))
            input_size = layer_size * 2

        self.label_decoder = lstm_builder(1, self.config.proj_size, self.config.label_rnn_size, self.model)

        self.proj_w1 = self.model.add_parameters((self.config.proj_size, input_size))
        self.proj_w2 = self.model.add_parameters((self.config.proj_size, input_size))
        self.proj_w3 = self.model.add_parameters((self.config.proj_size, input_size))
        self.proj_b1 = self.model.add_parameters((self.config.proj_size))
        self.proj_b2 = self.model.add_parameters((self.config.proj_size))
        self.proj_b3 = self.model.add_parameters((self.config.proj_size))

        self.label_w = dy.model.add_parameters((len(self.encodings.label2int), self.label_rnn_sizes))
        self.label_b = dy.model.add_parameters((len(self.encodings.label2int)))

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
        return []

    def save(self, path):
        print ("\tStoring " + path + "\n")
        self.model.save(path)

    def load(self, path):
        print ("\tLoading " + path + "\n")
        self.model.populate(path)

    def tag(self, seq):
        return false
