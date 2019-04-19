#
# Author: Tiberiu Boros
#
# Copyright (c) 2019 Adobe Systems Incorporated. All rights reserved.
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

# Code adapted from https://github.com/rguthrie3/BiLSTM-CRF/blob/master/model.py

import dynet as dy
import numpy as np


class CRFLabeler:
    def __init__(self, tagset_size, num_lstm_layers, hidden_dim, input_dim, model=None):
        if model is None:
            self.model = dy.Model()
        else:
            self.model = model
        self.tagset_size = tagset_size + 2

        self.START = tagset_size
        self.STOP = tagset_size + 1

        # LSTM parameters
        self.bi_lstm = dy.BiRNNBuilder(num_lstm_layers, input_dim, hidden_dim, self.model, dy.LSTMBuilder)

        # Matrix that maps from Bi-LSTM output to num tags
        self.lstm_to_tags_params = self.model.add_parameters((self.tagset_size, hidden_dim))
        self.lstm_to_tags_bias = self.model.add_parameters(self.tagset_size)
        self.mlp_out = self.model.add_parameters((self.tagset_size, self.tagset_size))
        self.mlp_out_bias = self.model.add_parameters(self.tagset_size)

        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((self.tagset_size, self.tagset_size))

    def set_dropout(self, p):
        self.bi_lstm.set_dropout(p)

    def disable_dropout(self):
        self.bi_lstm.disable_dropout()

    def build_tagging_graph(self, sentence):
        # embeddings = [self.word_rep(w) for w in sentence]
        embeddings = sentence

        lstm_out = self.bi_lstm.transduce(embeddings)

        H = self.lstm_to_tags_params.expr(update=True)
        Hb = self.lstm_to_tags_bias.expr(update=True)
        O = self.mlp_out.expr(update=True)
        Ob = self.mlp_out_bias.expr(update=True)
        # H = dy.parameter(self.lstm_to_tags_params)
        # Hb = dy.parameter(self.lstm_to_tags_bias)
        # O = dy.parameter(self.mlp_out)
        # Ob = dy.parameter(self.mlp_out_bias)
        scores = []
        for rep in lstm_out:
            score_t = O * dy.tanh(H * rep + Hb) + Ob
            scores.append(score_t)

        return scores

    def score_sentence(self, observations, tags):
        assert len(observations) == len(tags)
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [self.START] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(self.transitions[tags[i + 1]], tags[i]) + dy.pick(obs, tags[i + 1])
            score_seq.append(score.value())
        score = score + dy.pick(self.transitions[self.STOP], tags[-1])
        return score

    def viterbi_loss(self, observations, tags):
        # observations = self.build_tagging_graph(sentence)
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations)
        if viterbi_tags != tags:
            gold_score = self.score_sentence(observations, tags)
            return (viterbi_score - gold_score), viterbi_tags
        else:
            return dy.scalarInput(0), viterbi_tags

    def neg_log_loss(self, sentence, tags):
        observations = self.build_tagging_graph(sentence)
        gold_score = self.score_sentence(observations, tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score

    def forward(self, observations):

        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.tagset_size)
            return max_score_expr + dy.log(dy.sum_dim(dy.transpose(dy.exp(scores - max_score_expr_broadcast)), [1]))

        init_alphas = [-1e10] * self.tagset_size
        init_alphas[self.START] = 0
        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.tagset_size)
                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[self.STOP]
        alpha = log_sum_exp(terminal_expr)
        return alpha

    def viterbi_decoding(self, observations):
        backpointers = []
        init_vvars = [-1e10] * self.tagset_size
        init_vvars[self.START] = 0  # <Start> has all the probability
        for_expr = dy.inputVector(init_vvars)
        trans_exprs = [self.transitions[idx] for idx in range(self.tagset_size)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[self.STOP]
        terminal_arr = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id]  # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()  # Remove the start symbol
        best_path.reverse()
        assert start == self.START
        # Return best path and best path's score
        return best_path, path_score
