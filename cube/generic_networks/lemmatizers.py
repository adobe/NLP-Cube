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
import copy
import sys
from cube.misc.misc import fopen
from .character_embeddings import CharacterNetwork


class FSTLemmatizer:
    def __init__(self, config, encodings, embeddings, runtime=False):
        self.config = config
        self.encodings = encodings
        # Bug in encodings - will be removed after UD
        self.has_bug = False
        if self.encodings.char2int[' '] != 1:
            self.has_bug = True
            import sys
            sys.stdout.write("Detected encodings BUG!")
        self.embeddings = embeddings
        self.losses = []
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9, beta_2=0.9)
        self.character_network = CharacterNetwork(self.config.tag_embeddings_size, encodings,
                                                  rnn_size=self.config.char_rnn_size,
                                                  rnn_layers=self.config.char_rnn_layers,
                                                  embeddings_size=self.config.char_embeddings,
                                                  model=self.model, runtime=runtime)
        self.word2lemma = {}

        self.upos_lookup = self.model.add_lookup_parameters(
            (len(self.encodings.upos2int), self.config.tag_embeddings_size))
        self.xpos_lookup = self.model.add_lookup_parameters(
            (len(self.encodings.xpos2int), self.config.tag_embeddings_size))
        self.attrs_lookup = self.model.add_lookup_parameters(
            (len(self.encodings.attrs2int), self.config.tag_embeddings_size))
        self.char_lookup = self.model.add_lookup_parameters((len(self.encodings.char2int), self.config.char_embeddings))
        if runtime:
            self.rnn = dy.LSTMBuilder(self.config.rnn_layers,
                                      self.config.char_rnn_size * 2 + self.config.char_embeddings + self.config.tag_embeddings_size,
                                      self.config.rnn_size,
                                      self.model)
        else:
            from cube.generic_networks.utils import orthonormal_VanillaLSTMBuilder
            self.rnn = orthonormal_VanillaLSTMBuilder(self.config.rnn_layers,
                                                      self.config.char_rnn_size * 2 + self.config.char_embeddings + self.config.tag_embeddings_size,
                                                      self.config.rnn_size,
                                                      self.model)

        # self.att_w1 = self.model.add_parameters((200, self.config.char_rnn_size * 2))
        # self.att_w2 = self.model.add_parameters((200, self.config.rnn_size + self.config.tag_embeddings_size))
        # self.att_v = self.model.add_parameters((1, 200))

        self.start_lookup = self.model.add_lookup_parameters(
            (1, self.config.char_rnn_size * 2 + self.config.char_embeddings + self.config.tag_embeddings_size))

        self.softmax_w = self.model.add_parameters((len(self.encodings.char2int) + 3, self.config.rnn_size))
        self.softmax_b = self.model.add_parameters((len(self.encodings.char2int) + 3))

        ofs = len(self.encodings.char2int)
        self.label2int = {}
        self.label2int['<EOS>'] = ofs
        self.label2int['<COPY>'] = ofs + 1
        self.label2int['<INC>'] = ofs + 2

    def _attend(self, input_vectors, state, embeddings):
        w1 = self.att_w1.expr(update=True)
        w2 = self.att_w2.expr(update=True)
        v = self.att_v.expr(update=True)
        attention_weights = []

        w2dt = w2 * dy.concatenate([state.h()[-1], embeddings])
        for input_vector in input_vectors:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)

        attention_weights = dy.softmax(dy.concatenate(attention_weights))

        output_vectors = dy.esum(
            [vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])

        return output_vectors

    def _predict(self, word, upos, xpos, attrs, max_predictions=0, runtime=True, gs_labels=None):
        char_emb, states = self.character_network.compute_embeddings(word, runtime=runtime)

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
        num_predictions = 0
        i_src = 0
        i_labels = 0
        while num_predictions < max_predictions:
            # attention = self._attend(states, rnn, tag_emb)

            input = dy.concatenate([char_emb, states[i_src], tag_emb])
            rnn = rnn.add_input(input)

            softmax = dy.softmax(self.softmax_w.expr(update=True) * rnn.output() + self.softmax_b.expr(update=True))
            softmax_list.append(softmax)
            num_predictions += 1
            if runtime:
                l_index = np.argmax(softmax.npvalue())
                if l_index == self.label2int['<EOS>']:
                    break
                elif l_index == self.label2int['<INC>'] and i_src < len(states) - 1:
                    i_src += 1
            else:
                if gs_labels[i_labels] == '<INC>' and i_src < len(states) - 1:
                    i_src += 1
            i_labels += 1

        return softmax_list

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
        self.losses = []
        return total_loss

    def learn(self, seq):
        for entry in seq:
            if entry.upos != 'NUM' and entry.upos != 'PROPN':
                # print entry.word+"\t"+entry.lemma
                import sys
                if sys.version_info[0] == 2:
                    y_real = self._compute_transduction_states(unicode(entry.word, 'utf-8').lower(),
                                                               unicode(entry.lemma, 'utf-8').lower())
                else:
                    y_real = self._compute_transduction_states(entry.word.lower(),
                                                               entry.lemma.lower())
                # print y_real
                losses = []
                n_chars = len(y_real)
                # print entry.word, entry.lemma
                # print y_real
                softmax_output_list = self._predict(entry.word, entry.upos, entry.xpos, entry.attrs,
                                                    max_predictions=n_chars, runtime=False, gs_labels=y_real)
                # print unilemma.encode('utf-8')#, softmax_output_list
                for softmax, y_target in zip(softmax_output_list, y_real):
                    if y_target in self.label2int:
                        losses.append(-dy.log(dy.pick(softmax, self.label2int[y_target])))
                    elif y_target in self.encodings.char2int:
                        losses.append(-dy.log(dy.pick(softmax, self.encodings.char2int[y_target])))

                if len(losses) > 0:
                    loss = dy.esum(losses)

                self.losses.append(loss)

    def _compute_transduction_states(self, source, destination):
        a = np.zeros((len(source) + 1, len(destination) + 1))
        for i in range(len(source) + 1):
            a[i, 0] = i

        for i in range(len(destination) + 1):
            a[0, i] = i

        for i in range(1, len(source) + 1):
            for j in range(1, len(destination) + 1):
                cost = 0
                if source[i - 1] != destination[j - 1]:
                    cost = 1
                m = min([a[i - 1, j - 1], a[i - 1, j], a[i, j - 1]])
                a[i, j] = m + cost

        alignments = [-1] * len(destination)

        i = len(source)
        j = len(destination)
        while i > 1 or j > 1:
            if source[i - 1] == destination[j - 1]:
                alignments[j - 1] = i - 1
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                if a[i - 1, j - 1] <= a[i - 1, j] and a[i - 1, j - 1] <= a[i, j - 1]:
                    i -= 1
                    j -= 1
                elif a[i - 1][j] <= a[i - 1, j - 1] and a[i - 1, j] <= a[i, j - 1]:
                    i -= 1
                else:
                    j -= 1
        if source[i - 1] == destination[j - 1]:
            alignments[j - 1] = i - 1

        y_pred = []
        index_src = 0
        index_dst = 0
        while index_dst < len(destination):
            if alignments[index_dst] == index_src:
                y_pred.append("<COPY>")
                index_dst += 1
            elif alignments[index_dst] == -1:
                if destination[index_dst] == "\t":
                    y_pred.append("<TOK>")
                    index_dst += 1
                else:
                    y_pred.append(destination[index_dst])
                    index_dst += 1
            else:
                y_pred.append("<INC>")
                index_src += 1

        y_pred.append("<EOS>")
        return y_pred

    def tag(self, seq):
        dy.renew_cg()
        lemmas = []
        for entry in seq:
            if entry.upos == 'NUM' or entry.upos == 'PROPN':
                import sys
                if sys.version_info[0] == 2:

                    lemma = entry.word.decode('utf-8')
                else:
                    lemma = entry.word
            else:
                # check dictionary
                import sys
                if sys.version_info[0]==2:
                    key = entry.word.decode('utf-8').lower().encode('utf-8') + "\t" + entry.lemma
                else:
                    key = entry.word.lower() + "\t" + entry.lemma                    
                if key in self.word2lemma:
                    if sys.version_info[0]==2:
                        lemma=unicode(self.word2lemma[key],'utf-8')
                    else:                        
                        lemma=copy.deepcopy(self.word2lemma[key])
                else:
                    if sys.version_info[0]==2:
                        uniword = unicode(entry.word, 'utf-8')
                    else: 
                        uniword=copy.deepcopy(entry.word)
                        

                    softmax_output_list = self._predict(uniword, entry.upos, entry.xpos, entry.attrs,
                                                        max_predictions=500, runtime=True)
                    lemma = ""
                    src_index = 0
                    for softmax in softmax_output_list[:-1]:
                        label_index = np.argmax(softmax.npvalue())
                        if label_index == self.label2int['<COPY>'] and src_index < len(uniword):
                            lemma += uniword[src_index]
                        elif label_index == self.label2int['<INC>'] or label_index == self.label2int['<EOS>']:
                            src_index += 1
                        elif label_index < len(self.encodings.characters):
                            # if self.has_bug and label_index >= self.encodings.char2int[' ']:
                            #     label_index += 1
                            lemma += self.encodings.characters[label_index]
            # print entry.word+"\t"+lemma.encode('utf-8')
            if entry.upos != 'PROPN':
                lemmas.append(lemma.lower())
            else:
                lemmas.append(lemma)
        return lemmas

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.populate(path)
        dict_path = path.replace(".bestAcc", ".dict")
        import os.path
        if os.path.exists(dict_path):
            self.load_dict(dict_path)

    def load_dict(self, path):
        #print ("Loading lemma dictionary")
        with fopen(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 5:
                    if sys.version_info[0] == 2:
                        word = unicode(parts[0], 'utf-8').lower().encode('utf-8')
                    else:
                        word = parts[0].lower()
                    upos = parts[1]
                    key = word + '\t' + upos
                    self.word2lemma[key] = parts[4]
        #print ("Loaded " + str(len(self.word2lemma)) + " pairs")

    def lemmatize_sequences(self, sequences):
        new_sequences = []
        for sequence in sequences:
            new_sequence = copy.deepcopy(sequence)
            predicted_lemmas = self.tag(new_sequence)

            for entry, lemma in zip(new_sequence, predicted_lemmas):
                if not entry.is_compound_entry:
                    entry.lemma = lemma if lemma is not None else "_"  # lemma.encode('utf-8')
                else:
                    entry.lemma = "_"
            # for entryIndex, lemma in enumerate(predicted_lemmas):
            #    new_sequence[entryIndex].lemma = lemma if lemma is not None else "_"
            new_sequences.append(new_sequence)
        return new_sequences


class BDRNNLemmatizer:
    def __init__(self, lemmatizer_config, encodings, embeddings, runtime=False):
        self.config = lemmatizer_config
        self.encodings = encodings
        # Bug in encodings - this will be removed after UD Shared Task
        self.has_bug = False
        if self.encodings.char2int[' '] != 1:
            self.has_bug = True
        self.embeddings = embeddings
        self.losses = []

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
        self.char_lookup = self.model.add_lookup_parameters((len(self.encodings.char2int), self.config.char_embeddings))

        if runtime:
            self.rnn = dy.LSTMBuilder(self.config.rnn_layers,
                                      self.config.char_rnn_size * 2 + self.config.char_embeddings, self.config.rnn_size,
                                      self.model)
        else:
            from generic_networks.utils import orthonormal_VanillaLSTMBuilder
            self.rnn = orthonormal_VanillaLSTMBuilder(self.config.rnn_layers,
                                                      self.config.char_rnn_size * 2 + self.config.char_embeddings,
                                                      self.config.rnn_size,
                                                      self.model)

        self.att_w1 = self.model.add_parameters((200, self.config.char_rnn_size * 2))
        self.att_w2 = self.model.add_parameters((200, self.config.rnn_size + self.config.tag_embeddings_size))
        self.att_v = self.model.add_parameters((1, 200))

        self.start_lookup = self.model.add_lookup_parameters(
            (1, self.config.char_rnn_size * 2 + self.config.char_embeddings))

        self.softmax_w = self.model.add_parameters((len(self.encodings.char2int) + 1, self.config.rnn_size))
        self.softmax_b = self.model.add_parameters((len(self.encodings.char2int) + 1))
        self.softmax_casing_w = self.model.add_parameters((2, self.config.rnn_size))
        self.softmax_casing_b = self.model.add_parameters((2))

    def _attend(self, input_vectors, state, embeddings):
        w1 = self.att_w1.expr(update=True)
        w2 = self.att_w2.expr(update=True)
        v = self.att_v.expr(update=True)
        attention_weights = []

        w2dt = w2 * dy.concatenate([state.s()[-1], embeddings])
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
        char_emb = dy.inputVector([0] * self.config.char_embeddings)

        while True:
            attention = self._attend(states, rnn, tag_emb)

            input = dy.concatenate([attention, char_emb])
            rnn = rnn.add_input(input)

            softmax = dy.softmax(self.softmax_w.expr(update=True) * rnn.output() + self.softmax_b.expr(update=True))
            softmax_casing = dy.softmax(self.softmax_casing_w.expr(update=True) * rnn.output() + self.softmax_casing_b.expr(update=True))
            softmax_list.append([softmax, softmax_casing])
            if num_chars == 0:
                s_index = np.argmax(softmax.npvalue())
                if s_index == len(self.encodings.char2int):
                    break
                char_emb = self.char_lookup[s_index]
            else:
                if num_predictions < len(gs_chars):
                    char = gs_chars[num_predictions]
                    if char in self.encodings.char2int:
                        char_emb = self.char_lookup[self.encodings.char2int[char]]
                    else:
                        char_emb = self.char_lookup[self.encodings.char2int["<UNK>"]]

            num_predictions += 1
            if num_predictions == num_chars or num_predictions > 255:
                break

        return softmax_list

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
        self.losses = []
        return total_loss

    def learn(self, seq):

        for entry in seq:
            if entry.upos != 'NUM' and entry.upos != 'PROPN':
                losses = []
                unilemma = unicode(entry.lemma, 'utf-8')
                n_chars = len(unilemma)
                softmax_output_list = self._predict(entry.word, entry.upos, entry.xpos, entry.attrs,
                                                    num_chars=n_chars + 1, gs_chars=unilemma)
                # print unilemma.encode('utf-8')#, softmax_output_list
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
                    # print np.argmax(softmax[0].npvalue()), char_index, softmax

                losses.append(-dy.log(dy.pick(softmax_output_list[-1][0], len(self.encodings.char2int))))
                loss = dy.esum(losses)
                self.losses.append(loss)

    def tag(self, seq):
        dy.renew_cg()
        lemmas = []
        for entry in seq:
            if entry.upos == 'NUM' or entry.upos == 'PROPN':
                if sys.version_info[0] == 2:
                    lemma = entry.word.decode('utf-8')
                else:
                    lemma = entry.word
            else:
                softmax_output_list = self._predict(entry.word, entry.upos, entry.xpos, entry.attrs)
                lemma = ""
                for softmax in softmax_output_list[:-1]:
                    char_index = np.argmax(softmax[0].npvalue())
                    if char_index < len(self.encodings.characters):
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
