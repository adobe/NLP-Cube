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


class CompoundWordExpander:
    def __init__(self, config, encodings, embeddings, runtime=False):
        self.config = config
        self.word_embeddings = embeddings
        self.encodings = encodings
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9, beta_2=0.9)

        from cube.character_embeddings import CharacterNetwork
        self.encoder = CharacterNetwork(self.config.character_embeddings_size, encodings, self.config.encoder_size,
                                        self.config.encoder_layers, self.config.character_embeddings_size, self.model,
                                        runtime=runtime)

        self.decoder = dy.VanillaLSTMBuilder(self.config.decoder_layers, self.config.encoder_size * 2,
                                             self.config.decoder_size, self.model)
        self.decoder_start_lookup = self.model.add_lookup_parameters((1, self.config.encoder_size * 2))

        # self.att_w1 = self.model.add_parameters(
        #     (self.config.character_embeddings_size * 2, self.config.encoder_size * 2))
        # self.att_w2 = self.model.add_parameters(
        #     (self.config.character_embeddings_size * 2, self.config.decoder_size))
        # self.att_v = self.model.add_parameters((1, self.config.character_embeddings_size * 2))

        self.softmax_w = self.model.add_parameters(
            (len(self.encodings.char2int) + 4,
             self.config.decoder_size))  # all known characters except digits with COPY, INC, TOK and EOS
        self.softmax_b = self.model.add_parameters((len(self.encodings.char2int) + 4))

        self.softmax_comp_w = self.model.add_parameters((2, self.config.character_embeddings_size))
        self.softmax_comp_b = self.model.add_parameters((2))

        self.label2int = {}
        ofs = len(self.encodings.char2int)
        self.label2int['<EOS>'] = ofs
        self.label2int['<TOK>'] = ofs + 1
        self.label2int['<COPY>'] = ofs + 2
        self.label2int['<INC>'] = ofs + 3

        self.losses = []

    def start_batch(self):
        self.losses = []
        dy.renew_cg()

    def end_batch(self):
        total_loss = 0
        if len(self.losses) != 0:
            loss = dy.esum(self.losses)
            self.losses = []
            total_loss = loss.value()
            loss.backward()
            self.trainer.update()
        dy.renew_cg()
        return total_loss

    def learn(self, seq):
        losses = []
        examples = self._get_examples(seq)

        for example in examples:
            y_pred, encoder_states = self._predict_is_compound_entry(example.source, runtime=False)
            if not example.should_expand:
                losses.append(-dy.log(dy.pick(y_pred, 0)))
            else:
                losses.append(-dy.log(dy.pick(y_pred, 1)))
                losses.append(self._learn_transduction(example.source, example.destination, encoder_states))
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

    def _decode(self, encoder_states, runtime=True, max_preds=-1, gs_labels=None):
        y_pred = []
        num_preds = 0
        lstm = self.decoder.initial_state().add_input(self.decoder_start_lookup[0])
        i_src = 0
        while num_preds < max_preds:
            # input = self._attend(encoder_states, lstm)
            input = encoder_states[i_src]
            lstm = lstm.add_input(input)
            softmax_out = dy.softmax(self.softmax_w.expr(update=True) * lstm.output() + self.softmax_b.expr(update=True))
            y_pred.append(softmax_out)

            if runtime:
                if np.argmax(softmax_out.npvalue()) == self.label2int['<EOS>']:
                    return y_pred
                elif np.argmax(softmax_out.npvalue()) == self.label2int['<INC>'] and i_src < len(encoder_states) - 1:
                    i_src += 1
            else:
                if gs_labels[num_preds] == '<INC>' and i_src < len(encoder_states) - 1:
                    i_src += 1
            num_preds += 1
        return y_pred

    def _learn_transduction(self, source, destination, encoder_states):
        losses = []
        y_target = self._compute_transduction_states(source, destination)
        y_predicted = self._decode(encoder_states, runtime=False, max_preds=len(y_target), gs_labels=y_target)
        for y_real, y_pred in zip(y_target, y_predicted):
            if y_real in self.label2int:
                losses.append(-dy.log(dy.pick(y_pred, self.label2int[y_real])))
            else:
                if y_real in self.encodings.char2int:
                    losses.append(-dy.log(dy.pick(y_pred, self.encodings.char2int[y_real])))
                # else:
                #    print source + "\t\t" + destination

        return dy.esum(losses)

    def _predict_is_compound_entry(self, word, runtime=True):
        emb, states = self.encoder.compute_embeddings(word, runtime=runtime)
        output = dy.softmax(self.softmax_comp_w.expr(update=True) * emb + self.softmax_comp_b.expr(update=True))
        return output, states

    def _transduce(self, source, encoder_states):
        tokens = []
        y_pred = self._decode(encoder_states, runtime=True, max_preds=100)

        i_src = 0
        token = ""
        for y in y_pred:
            y = np.argmax(y.npvalue())
            if y == self.label2int['<INC>'] and i_src < len(encoder_states) - 1:
                i_src += 1
            elif y == self.label2int['<COPY>']:
                if i_src < len(source):
                    token += source[i_src]
            elif y == self.label2int['<TOK>'] or y == self.label2int['<EOS>']:
                tokens.append(token)
                token = ""
            else:
                if y < len(self.encodings.characters):
                    token += self.encodings.characters[y]

        return tokens

    def tag_token(self, word):
        dy.renew_cg()
        compound = False
        word = unicode(word, 'utf-8')
        tokens = []
        ce_out, encoder_states = self._predict_is_compound_entry(word, runtime=True)
        if np.argmax(ce_out.npvalue()) == 1:
            tokens = self._transduce(word, encoder_states)
            compound = True
        return compound, tokens

    def tag(self, seq):
        dy.renew_cg()
        new_seq = []
        index = 1
        for entry in seq:
            if not entry.is_compound_entry:
                ce_out, encoder_states = self._predict_is_compound_entry(unicode(entry.word, 'utf-8'), runtime=True)
                if np.argmax(ce_out.npvalue()) == 0:
                    entry.index = index
                    new_seq.append(entry)
                    index += 1
                else:
                    compounds = self._transduce(unicode(entry.word, 'utf-8'), encoder_states)
                    # bug because _transduce may return empty tokens
                    valid_tokens = []
                    for token in compounds:
                        if token.strip() != "":
                            valid_tokens.append(token)
                    compounds = valid_tokens
                    if len(compounds) <= 1:
                        entry.index = index
                        new_seq.append(entry)
                        index += 1
                    else:
                        entry.index = str(index) + '-' + str(index + len(compounds) - 1)
                        entry.is_compound_entry = True
                        entry.upos='_'
                        entry.xpos = '_'
                        entry.attrs = '_'
                        entry.label = '_'
                        entry.head = '_'
                        entry.deps = '_'
                        new_seq.append(entry)
                        for word in compounds:
                            from io_utils.conll import ConllEntry
                            entry = ConllEntry(index, word.encode('utf-8'), word.encode('utf-8'), '_', '_', '_', '0',
                                               '_',
                                               '_',
                                               '')
                            new_seq.append(entry)
                            index += 1

        return new_seq

    def _get_examples(self, seq):
        examples = []
        cww = 0
        for entry in seq:
            if cww == 0:
                et = ExpandedToken(source=unicode(entry.word, 'utf-8'))
                if entry.is_compound_entry:
                    et.should_expand = True
                    et.destination = u''
                    interval = entry.index
                    interval = interval.split("-")
                    stop = int(interval[1])
                    start = int(interval[0])
                    cww = stop - start + 1
                else:
                    et.destination = et.source
                    examples.append(et)
            else:
                et.destination += "\t" + unicode(entry.word, 'utf-8')
                cww -= 1
                if cww == 0:
                    et.destination = et.destination.strip()
                    examples.append(et)

        return examples

    def save(self, filename):
        self.model.save(filename)

    def load(self, path):
        self.model.populate(path)

    def expand_sequences(self, sequences):
        new_sequences = []
        for sequence in sequences:
            new_sequence = self.tag(sequence)
            new_sequences.append(new_sequence)
        return new_sequences


class ExpandedToken:
    def __init__(self, source=None, destination=None, should_expand=False):
        self.source = source
        self.destination = destination
        self.should_expand = should_expand
