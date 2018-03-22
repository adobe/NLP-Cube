import dynet as dy
import numpy as np


class CompoundWordExpander:
    def __init__(self, config, encodings, embeddings, runtime=False):
        self.config = config
        self.word_embeddings = embeddings
        self.encodings = encodings
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9, beta_2=0.9)

        from character_embeddings import CharacterNetwork
        self.encoder = CharacterNetwork(self.config.character_embeddings_size, encodings, self.config.encoder_size,
                                        self.config.encoder_layers, self.config.character_embeddings_size, self.model,
                                        runtime=runtime)

        self.decoder = dy.VanillaLSTMBuilder(self.config.decoder_layers, self.config.character_embeddings_size * 2,
                                             self.config.decoder_size, self.model)

        self.att_w1 = self.model.add_parameters(
            (self.config.character_embeddings_size * 2, self.config.character_embeddings_size * 2))
        self.att_w2 = self.model.add_parameters(
            (self.config.character_embeddings_size * 2, self.config.decoder_size))
        self.att_v = self.model.add_parameters((1, self.config.character_embeddings_size * 2))

        self.softmax_w = self.model.add_parameters(
            (len(self.encodings.char2int) + 3,
             self.config.decoder_size))  # all known characters except digits with COPY, INC and EOS
        self.softmax_b = self.model.add_parameters((len(self.encodings.char2int) + 3))

        self.softmax_comp_w = self.model.add_parameters((2, self.config.character_embeddings_size))
        self.softmax_comp_b = self.model.add_parameters((2))

        self.losses = []

    def start_batch(self):
        self.losses = []
        dy.renew_cg()

    def end_batch(self):
        loss = dy.esum(self.losses)
        self.losses = []
        total_loss = loss.value()
        loss.backward()
        self.trainer.update()

    def learn(self, seq):
        losses = []
        examples = self._get_examples(seq)

        for example in examples:
            y_pred, encoder_states = self._predict_is_compound_entry(example.source)
            if not example.isCompoundEntry:
                losses.append(-dy.log(dy.pick(y_pred, 0)))
            else:
                losses.append(-dy.log(dy.pick(y_pred, 1)))
                self._learn_transduction(example.source, example.destination, encoder_states)

    def _get_examples(self, seq):
        cww = 0
