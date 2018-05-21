import copy
import dynet as dy
import numpy as np

from character_embeddings import CharacterNetwork


def get_link(seq, iSrc, iDst):
    l1 = seq[iSrc].label
    l2 = seq[iDst].label

    if iSrc == 0 and l2 != '*':
        return 1
    if iDst == 0 and l1 != '*':
        return 1

    if l1 == "*" or l2 == "*":
        return 0



    if (l2 in l1) or (l1 in l2):
        # print l1, l2, 1
        return 1
    else:
        # print l1, l2, 0
        return 0


class Network(object):

    def __init__(self, config, encodings):
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9, beta_2=0.9)
        self.config = config
        self.encodings = encodings
        self.proj_size = 100

        self.word_lookup = self.model.add_lookup_parameters((len(self.encodings.word2int), self.proj_size))
        self.lemma_lookup = self.model.add_lookup_parameters((len(self.encodings.lemma2int), self.proj_size))
        self.UPOS_lookup = self.model.add_lookup_parameters((len(self.encodings.UPOS2int), self.proj_size))
        self.XPOS_lookup = self.model.add_lookup_parameters((len(self.encodings.XPOS2int), self.proj_size))
        self.attrs_lookup = self.model.add_lookup_parameters((len(self.encodings.attrs2int), self.proj_size))

        self.encoder_fw = []
        self.encoder_bw = []

        self.encoder_fw.append(dy.LSTMBuilder(1, self.proj_size * 2, config.encoder_size, self.model))
        self.encoder_bw.append(dy.LSTMBuilder(1, self.proj_size * 2, config.encoder_size, self.model))

        [self.encoder_fw.append(dy.LSTMBuilder(1, self.config.encoder_size * 2, self.config.encoder_size, self.model))
         for _ in range(self.config.encoder_layers - 1)]
        [self.encoder_bw.append(dy.LSTMBuilder(1, self.config.encoder_size * 2, self.config.encoder_size, self.model))
         for _ in range(self.config.encoder_layers - 1)]

        self.proj_w1 = self.model.add_parameters((config.proj_size, config.encoder_size * 2))
        self.proj_w2 = self.model.add_parameters((config.proj_size, config.encoder_size * 2))
        self.proj_w3 = self.model.add_parameters((config.proj_size, config.encoder_size * 2))
        self.proj_b1 = self.model.add_parameters((config.proj_size))
        self.proj_b2 = self.model.add_parameters((config.proj_size))
        self.proj_b3 = self.model.add_parameters((config.proj_size))

        self.linkW = self.model.add_parameters((2, config.proj_size * 2))
        self.linkB = self.model.add_parameters((2))

        self.label_lstm = dy.LSTMBuilder(config.label_encoder_layers, config.proj_size, config.label_encoder_size,
                                         self.model)
        self.labelW = self.model.add_parameters((len(encodings.label2int), config.label_encoder_size))
        self.labelB = self.model.add_parameters((len(encodings.label2int)))
        self.losses = []

    def _make_input(self, sequence, runtime=True):
        all_input = []
        for entry in sequence:
            word = entry.word.decode('utf-8').lower().encode('utf-8')
            if word in self.encodings.word2int:
                word_vector = self.word_lookup[self.encodings.word2int[word]]
            else:
                word_vector = self.word_lookup[0]
            if entry.lemma in self.encodings.lemma2int:
                lemma_vector = self.lemma_lookup[self.encodings.lemma2int[entry.lemma]]
            else:
                lemma_vector = self.lemma_lookup[0]
            lex_vector = dy.tanh(word_vector + lemma_vector)

            morph_missing = 3
            zeroVec = dy.inputVector([0] * self.proj_size)
            UPOS_vector = None
            if entry.UPOS in self.encodings.UPOS2int:
                UPOS_vector = self.UPOS_lookup[self.encodings.UPOS2int[entry.UPOS]]
                morph_missing -= 1
            else:
                UPOS_vector = zeroVec

            XPOS_vector = None
            if entry.XPOS in self.encodings.XPOS2int:
                XPOS_vector = self.XPOS_lookup[self.encodings.XPOS2int[entry.XPOS]]
                morph_missing -= 1
            else:
                XPOS_vector = zeroVec

            attrs_vector = None
            if entry.attrs in self.encodings.attrs2int:
                attrs_vector = self.attrs_lookup[self.encodings.attrs2int[entry.attrs]]
                morph_missing -= 1
            else:
                attrs_vector = zeroVec

            # implicit nu lipseste nici un vector (0)
            padding_factor = 1
            if morph_missing == 1:
                padding_factor = 1.5
            elif morph_missing == 2:
                padding_factor = 3

            if padding_factor is not None:
                morph_vector = dy.tanh((UPOS_vector + XPOS_vector + attrs_vector) * padding_factor)

            if runtime:
                sequence_input = dy.concatenate([lex_vector, morph_vector])
            else:
                p1 = np.random.random()
                p2 = np.random.random()
                scale = 1
                if p1 < 0.34:
                    morph_vector = zeroVec
                    scale = 2
                if p2 < 0.34:
                    lex_vector = zeroVec
                    scale = 2
                scale = dy.scalarInput(scale)
                sequence_input = dy.concatenate([lex_vector, morph_vector]) * scale

            all_input.append(sequence_input)
        return all_input

    def predict(self, seq, runtime=True):
        x_list = self._make_input(seq, runtime=runtime)

        for fw, bw in zip(self.encoder_fw, self.encoder_bw):
            x_fw = fw.initial_state().transduce(x_list)
            x_bw = list(reversed(bw.initial_state().transduce(reversed(x_list))))
            x_list = [dy.concatenate([x1, x2]) for x1, x2 in zip(x_fw, x_bw)]

        proj_x1 = [dy.tanh(self.proj_w1.expr() * x + self.proj_b1.expr()) for x in x_list]
        proj_x2 = [dy.tanh(self.proj_w2.expr() * x + self.proj_b2.expr()) for x in x_list]
        proj_x3 = [dy.tanh(self.proj_w3.expr() * x + self.proj_b3.expr()) for x in x_list]

        output = []
        for iSrc in range(len(seq)):
            out_row = []
            for iDst in range(len(seq)):
                if iDst > iSrc:
                    x = dy.concatenate([proj_x1[iSrc], proj_x2[iDst]])
                    out_row.append(dy.softmax(self.linkW.expr() * x + self.linkB.expr()))
                else:
                    out_row.append(None)
            output.append(out_row)
        return output, proj_x3

    def _has_index(self, index, label):
        parts = label.split(";")
        for part in parts:
            pp = part.split(":")
            if pp[0] == str(index):
                return True
        return False

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
                if self._has_index(index, row.label):
                    if first:
                        first = False
                        parts = row.label.split(";")
                        for part in parts:
                            pp = part.split(":")
                            if pp[0] == str(index):
                                label = pp[1]
                                break

                    lst.append(i)
                i += 1
            if label == "":
                for row in seq:
                    print row.orig_line
            chains.append(lst)
            labels.append(label)

        return chains, labels

    def start_batch(self):
        self.losses = []
        dy.renew_cg()

    def end_batch(self):
        if len(self.losses) != 0:
            loss = dy.esum(self.losses)
            loss_val = loss.value()
            loss.backward()
            self.trainer.update()
            return loss_val
        else:
            return 0

    def learn(self, seq):
        output, proj_x3 = self.predict(seq, runtime=False)

        # arcs
        for iSrc in range(len(seq)):
            for iDst in range(len(seq)):
                if iDst > iSrc:
                    o = output[iSrc][iDst]  # the softmax portion
                    t = get_link(seq, iSrc, iDst)
                    # if t==1:
                    self.losses.append(-dy.log(dy.pick(o, t)))

        # labels
        gs_chains, labels = self._get_gs_chains(seq)

        for chain, label in zip(gs_chains, labels):
            label_rnn = self.label_lstm.initial_state()
            for index in chain:
                label_rnn = label_rnn.add_input(proj_x3[index])
            label_softmax = dy.softmax(self.labelW.expr() * label_rnn.output() + self.labelB.expr())
            self.losses.append(-dy.log(dy.pick(label_softmax, self.encodings.label2int[label])))

    def save_network(self, path):
        print "Creating", path
        self.model.save(path)

    def load_network(self, path):
        print "Restoring", path
        self.model.populate(path)

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
            solutions.append(copy.deepcopy(current_nodes))

    def decode(self, output, proj_x):
        expressions = []
        labels = []
        a = np.zeros((len(output), len(output)))
        for iSrc in range(len(output)):
            for iDst in range(len(output)):
                if iDst > iSrc:
                    if output[iSrc][iDst].value()[1] > output[iSrc][iDst].value()[0]:
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
            lstm_label = self.label_lstm.initial_state()
            for index in expression:
                lstm_label = lstm_label.add_input(proj_x[index])
            label_soft = self.labelW.expr() * lstm_label.output() + self.labelB.expr()
            label_index = np.argmax(label_soft.npvalue())
            labels.append(self.encodings.label_list[label_index])

        return expressions, labels
