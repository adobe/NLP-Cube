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
# and from https://github.com/neulab/cmu-ner/blob/master/models/decoders.py

import dynet as dy
import numpy as np


class CRFDecoder:
    def __init__(self, model, src_output_dim, tag_emb_dim, tag_size, constraints=None):
        self.model = model
        self.start_id = tag_size
        self.end_id = tag_size + 1
        self.tag_size = tag_size + 2
        tag_size = tag_size + 2

        # optional: transform the hidden space of src encodings into the tag embedding space
        self.W_src2tag_readout = model.add_parameters((tag_emb_dim, src_output_dim))
        self.b_src2tag_readout = model.add_parameters((tag_emb_dim))
        self.b_src2tag_readout.zero()

        self.W_scores_readout2tag = model.add_parameters((tag_size, tag_emb_dim))
        self.b_scores_readout2tag = model.add_parameters((tag_size))
        self.b_scores_readout2tag.zero()

        # (to, from), trans[i] is the transition score to i
        init_transition_matrix = np.random.randn(tag_size, tag_size)  # from, to
        # init_transition_matrix[self.start_id, :] = -1000.0
        # init_transition_matrix[:, self.end_id] = -1000.0
        init_transition_matrix[self.end_id, :] = -1000.0
        init_transition_matrix[:, self.start_id] = -1000.0
        if constraints is not None:
            init_transition_matrix = self._constrained_transition_init(init_transition_matrix, constraints)
        # print init_transition_matrix
        self.transition_matrix = model.lookup_parameters_from_numpy(init_transition_matrix)

        self.interpolation = True  # args.interp_crf_score
        if self.interpolation:
            self.W_weight_transition = model.add_parameters((1, tag_emb_dim))
        self.b_weight_transition = model.add_parameters((1))
        self.b_weight_transition.zero()

    def learn(self, src_enc, tgt_tags):
        return self.decode_loss(src_enc, [tgt_tags])

    def tag(self, src_enc):
        return self.decoding(src_enc)[1]

    def _constrained_transition_init(self, transition_matrix, contraints):
        '''
        :param transition_matrix: numpy array, (from, to)
        :param contraints: [[from_indexes], [to_indexes]]
        :return: newly initialized transition matrix
        '''
        for cons in contraints:
            transition_matrix[cons[0], cons[1]] = -1000.0
        return transition_matrix

    def _log_sum_exp_dim_0(self, x):
        # numerically stable log_sum_exp
        dims = x.dim()
        max_score = dy.max_dim(x, 0)  # (dim_1, batch_size)
        if len(dims[0]) == 1:
            max_score_extend = max_score
        else:
            max_score_reshape = dy.reshape(max_score, (1, dims[0][1]), batch_size=dims[1])
            max_score_extend = dy.concatenate([max_score_reshape] * dims[0][0])
        x = x - max_score_extend
        exp_x = dy.exp(x)
        # (dim_1, batch_size), if no dim_1, return ((1,), batch_size)
        log_sum_exp_x = dy.log(dy.mean_dim(exp_x, d=[0], b=False) * dims[0][0])
        return log_sum_exp_x + max_score

    def forward_alg(self, tag_scores):
        ''' Forward DP for CRF.
        tag_scores (list of batched dy.Tensor): (tag_size, batchsize)
        '''
        # Be aware: if a is lookup_parameter with 2 dimension, then a[i] returns one row;
        # if b = dy.parameter(a), then b[i] returns one column; which means dy.parameter(a) already transpose a
        transpose_transition_score = self.transition_matrix#.expr(update=True)
        # transpose_transition_score = dy.transpose(transition_score)
        # alpha(t', s) = the score of sequence from t=0 to t=t' in log space
        # np_init_alphas = -100.0 * np.ones((self.tag_size, batch_size))
        # np_init_alphas[self.start_id, :] = 0.0
        # alpha_tm1 = dy.inputTensor(np_init_alphas, batched=True)

        alpha_tm1 = transpose_transition_score[self.start_id] + tag_scores[0]
        # self.transition_matrix[i]: from i, column
        # transpose_score[i]: to i, row
        # transpose_score: to, from

        for tag_score in tag_scores[1:]:
            # extend for each transit <to>
            alpha_tm1 = dy.concatenate_cols([alpha_tm1] * self.tag_size)  # (from, to, batch_size)
            # each column i of tag_score will be the repeated emission score to tag i
            tag_score = dy.transpose(dy.concatenate_cols([tag_score] * self.tag_size))
            alpha_t = alpha_tm1 + transpose_transition_score + tag_score
            alpha_tm1 = self._log_sum_exp_dim_0(alpha_t)  # (tag_size, batch_size)

        terminal_alpha = self._log_sum_exp_dim_0(alpha_tm1 + self.transition_matrix[self.end_id])  # (1, batch_size)
        return terminal_alpha

    def score_one_sequence(self, tag_scores, tags, batch_size):
        ''' tags: list of tag ids at each time step '''
        # print tags, batch_size
        # print batch_size
        # print "scoring one sentence"
        tags = [[self.start_id] * batch_size] + tags  # len(tag_scores) = len(tags) - 1
        score = dy.inputTensor(np.zeros(batch_size), batched=True)
        # tag_scores = dy.concatenate_cols(tag_scores) # tot_tags, sent_len, batch_size
        # print "tag dim: ", tag_scores.dim()
        for i in range(len(tags) - 1):
            score += dy.pick_batch(dy.lookup_batch(self.transition_matrix, tags[i + 1]), tags[i]) \
                     + dy.pick_batch(tag_scores[i], tags[i + 1])
        score += dy.pick_batch(dy.lookup_batch(self.transition_matrix, [self.end_id] * batch_size), tags[-1])
        return score

    def _transpose_input(self, seq, padding_token=0):
        # input seq: list of samples [[w1, w2, ..], [w1, w2, ..]]
        max_len = max([len(sent) for sent in seq])
        seq_pad = []
        seq_mask = []
        for i in range(max_len):
            pad_temp = [sent[i] if i < len(sent) else padding_token for sent in seq]
            mask_temp = [1.0 if i < len(sent) else 0.0 for sent in seq]
            seq_pad.append(pad_temp)
            seq_mask.append(mask_temp)

        return seq_pad, seq_mask

    def decode_loss(self, src_encodings, tgt_tags):
        # This is the batched version which requires bucketed batch input with the same length.
        '''
        The length of src_encodings and tgt_tags are time_steps.
        src_encodings: list of dynet.Tensor (src_output_dim, batch_size)
        tgt_tags: list of tag ids [(1, batch_size)]
        return: average of negative log likelihood
        '''
        # TODO: transpose tgt tags first
        batch_size = len(tgt_tags)
        tgt_tags, tgt_mask = self._transpose_input(tgt_tags, 0)
        W_src2tag_readout = self.W_src2tag_readout.expr(update=True)
        b_src2tag_readout = self.b_src2tag_readout.expr(update=True)
        W_score_tag = self.W_scores_readout2tag.expr(update=True)
        b_score_tag = self.b_scores_readout2tag.expr(update=True)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding])) for src_encoding
                    in src_encodings]
        if self.interpolation:
            W_transit = self.W_weight_transition.expr(update=True)
            b_transit = self.b_weight_transition.expr(update=True)
            step_weight_on_transit = [dy.logistic(dy.affine_transform([b_transit, W_transit, tag_emb])) for tag_emb in
                                      tag_embs]

        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        # scores over all paths, all scores are in log-space
        forward_scores = self.forward_alg(tag_scores)
        gold_score = self.score_one_sequence(tag_scores, tgt_tags, batch_size)
        # negative log likelihood
        loss = dy.sum_batches(forward_scores - gold_score) / batch_size
        return loss  # , dy.sum_batches(forward_scores)/batch_size, dy.sum_batches(gold_score) / batch_size

    def get_crf_scores(self, src_encodings):
        W_src2tag_readout = self.W_src2tag_readout.expr(update=True)
        b_src2tag_readout = self.b_src2tag_readout.expr(update=True)
        W_score_tag = self.W_scores_readout2tag.expr(update=True)
        b_score_tag = self.b_scores_readout2tag.expr(update=True)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding]))
                    for src_encoding in src_encodings]
        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        transpose_transition_score = self.transition_matrix#.expr(update=True)  # (to, from)

        return transpose_transition_score.npvalue(), [ts.npvalue() for ts in tag_scores]

    def decoding(self, src_encodings):
        ''' Viterbi decoding for a single sequence. '''
        W_src2tag_readout = self.W_src2tag_readout.expr(update=True)
        b_src2tag_readout = self.b_src2tag_readout.expr(update=True)
        W_score_tag = self.W_scores_readout2tag.expr(update=True)
        b_score_tag = self.b_scores_readout2tag.expr(update=True)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding]))
                    for src_encoding in src_encodings]
        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        back_trace_tags = []
        np_init_alpha = np.ones(self.tag_size) * -2000.0
        np_init_alpha[self.start_id] = 0.0
        max_tm1 = dy.inputTensor(np_init_alpha)
        transpose_transition_score = self.transition_matrix#.expr(update=True) # (to, from)

        for i, tag_score in enumerate(tag_scores):
            max_tm1 = dy.concatenate_cols([max_tm1] * self.tag_size)
            max_t = max_tm1 + transpose_transition_score
            if i != 0:
                eval_score = max_t.npvalue()[:-2, :]
            else:
                eval_score = max_t.npvalue()
            best_tag = np.argmax(eval_score, axis=0)
            back_trace_tags.append(best_tag)
            max_tm1 = dy.inputTensor(eval_score[best_tag, range(self.tag_size)]) + tag_score

        terminal_max_T = max_tm1 + self.transition_matrix[self.end_id]
        eval_terminal = terminal_max_T.npvalue()[:-2]
        best_tag = np.argmax(eval_terminal, axis=0)
        best_path_score = eval_terminal[best_tag]

        best_path = [best_tag]
        for btpoint in reversed(back_trace_tags):
            best_tag = btpoint[best_tag]
            best_path.append(best_tag)
        start = best_path.pop()
        assert start == self.start_id
        best_path.reverse()
        return best_path_score, best_path

    def cal_accuracy(self, pred_path, true_path):
        return np.sum(np.equal(pred_path, true_path).astype(np.float32)) / len(pred_path)


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

    def learn(self, input, tags):
        return self._neg_log_loss(input, tags)

    def tag(self, input):
        return self._viterbi_decoding(self._build_tagging_graph(input))[0]

    def set_dropout(self, p):
        self.bi_lstm.set_dropout(p)

    def disable_dropout(self):
        self.bi_lstm.disable_dropout()

    def _build_tagging_graph(self, sentence):
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

    def _score_sentence(self, observations, tags):
        assert len(observations) == len(tags)
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [self.START] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(self.transitions[tags[i + 1]], tags[i]) + dy.pick(obs, tags[i + 1])
            score_seq.append(score.value())
        score = score + dy.pick(self.transitions[self.STOP], tags[-1])
        return score

    def _viterbi_loss(self, observations, tags):
        # observations = self.build_tagging_graph(sentence)
        viterbi_tags, viterbi_score = self._viterbi_decoding(observations)
        if viterbi_tags != tags:
            gold_score = self._score_sentence(observations, tags)
            return (viterbi_score - gold_score), viterbi_tags
        else:
            return dy.scalarInput(0), viterbi_tags

    def _neg_log_loss(self, sentence, tags):
        observations = self._build_tagging_graph(sentence)
        gold_score = self._score_sentence(observations, tags)
        forward_score = self._forward(observations)
        return forward_score - gold_score

    def _forward(self, observations):

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

    def _viterbi_decoding(self, observations):
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
