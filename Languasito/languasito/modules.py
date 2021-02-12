import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np


class LinearNorm(pl.LightningModule):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_normal_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_normal_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class WordGram(pl.LightningModule):
    def __init__(self, num_chars: int, num_langs: int, num_filters=512, char_emb_size=256, case_emb_size=32,
                 lang_emb_size=32, num_layers=3):
        super(WordGram, self).__init__()
        NUM_FILTERS = num_filters
        self._num_filters = NUM_FILTERS
        self._lang_emb = nn.Embedding(num_langs + 1, lang_emb_size)
        self._tok_emb = nn.Embedding(num_chars + 1, char_emb_size)
        self._case_emb = nn.Embedding(4, case_emb_size)
        self._num_layers = num_layers
        convolutions_char = []
        cs_inp = char_emb_size + lang_emb_size + case_emb_size
        for _ in range(num_layers):
            conv_layer = nn.Sequential(
                ConvNorm(cs_inp,
                         NUM_FILTERS,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(NUM_FILTERS))
            convolutions_char.append(conv_layer)
            cs_inp = NUM_FILTERS // 2 + lang_emb_size
        self._convolutions_char = nn.ModuleList(convolutions_char)
        self._pre_out = LinearNorm(NUM_FILTERS // 2, NUM_FILTERS // 2)

    def forward(self, x_char, x_case, x_lang, x_mask, x_word_len):
        x_char = self._tok_emb(x_char)
        x_case = self._case_emb(x_case)
        x_lang = self._lang_emb(x_lang)

        x = torch.cat([x_char, x_case], dim=-1)
        x = x.permute(0, 2, 1)
        x_lang = x_lang.unsqueeze(1).repeat(1, x_case.shape[1], 1).permute(0, 2, 1)
        half = self._num_filters // 2
        count = 0
        res = None
        skip = None
        for conv in self._convolutions_char:
            count += 1
            drop = self.training
            if count >= len(self._convolutions_char):
                drop = False
            if skip is not None:
                x = x + skip

            x = torch.cat([x, x_lang], dim=1)
            conv_out = conv(x)
            tmp = torch.tanh(conv_out[:, :half, :]) * torch.sigmoid((conv_out[:, half:, :]))
            if res is None:
                res = tmp
            else:
                res = res + tmp
            skip = tmp
            x = torch.dropout(tmp, 0.1, drop)
        x = x + res
        x = x.permute(0, 2, 1)
        x = x * x_mask.unsqueeze(2)
        pre = torch.sum(x, dim=1, dtype=torch.float)
        norm = pre / x_word_len.unsqueeze(1)
        # embeds = self._pre_out(norm)
        # norm = embeds.norm(p=2, dim=-1, keepdim=True)
        # embeds_normalized = embeds.div(norm)
        # return embeds_normalized

        return torch.tanh(self._pre_out(norm))

    def _get_device(self):
        if self._lang_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._lang_emb.weight.device.type, str(self._lang_emb.weight.device.index))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


class GE2ELoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax'):
        '''
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        '''
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ['softmax', 'contrast']

        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        '''
        Calculates the new centroids excluding the reference utterance
        '''
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        '''
        Make the cosine similarity matrix with dims (N,M,N)
        '''
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
                # vector based cosine similarity for speed
                cs_row.append(torch.clamp(
                    torch.mm(utterance.unsqueeze(1).transpose(0, 1), new_centroids.transpose(0, 1)) / (
                            torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6))
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-torch.nn.functional.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j + 1:]))
                L_row.append(1. - torch.sigmoid(cos_sim_matrix[j, i, j]) + torch.max(excl_centroids_sigmoids))
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        '''
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        '''
        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.mean()


class WordDecoder(nn.Module):
    def __init__(self, cond_size: int, char_emb_size: int, vocab_size: int, rnn_size: int = 200, rnn_layers: int = 2):
        super().__init__()
        self._char_emb_size = char_emb_size
        self._vocab_size = vocab_size
        self._cond_size = cond_size

        self._char_emb = nn.Embedding(vocab_size, char_emb_size)
        self._rnn = nn.LSTM(cond_size + char_emb_size, rnn_size, num_layers=rnn_layers, batch_first=True)
        self._output = nn.Linear(rnn_size, vocab_size)

    def forward(self, cond, gs_chars=None):
        if gs_chars is not None:
            cond = cond.unsqueeze(1).repeat(1, gs_chars.shape[1], 1)
            gs_chars = self._char_emb(gs_chars)
            x_input = torch.cat([cond, gs_chars], dim=-1)
            x_out_rnn, _ = self._rnn(x_input)
            return self._output(x_out_rnn)[:, :-1, :]
        else:
            reached_end = [False for ii in range(cond.shape[0])]
            last_char = np.ones((cond.shape[0], 1)) * 2  # <SOT>
            last_char = torch.tensor(last_char, dtype=torch.long, device=self._get_device())
            last_char = self._char_emb(last_char)
            cond = cond.unsqueeze(1)
            index = 0
            decoder_hidden = None
            output_list = []
            while True:
                decoder_input = torch.cat([cond, last_char], dim=-1)
                decoder_output, decoder_hidden = self._rnn(decoder_input, hx=decoder_hidden)
                output = self._output(decoder_output)
                last_char = torch.argmax(output, dim=-1)
                output = last_char.detach().cpu().numpy()
                for ii in range(output.shape[0]):
                    if output[ii] == 3:  # <EOT>
                        reached_end[ii] = True

                output_list.append(last_char.detach().cpu())
                last_char = self._char_emb(last_char)

                index += 1
                if np.all(reached_end):
                    break
            output = torch.cat(output_list, dim=1).detach().cpu().numpy()
            return output

    def _get_device(self):
        if self._char_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._char_emb.weight.device.type, str(self._char_emb.weight.device.index))


def log1pexp(x):
    return torch.where(x < 50, torch.log1p(torch.exp(x)), x)


import random


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._cs_loss = torch.nn.CosineEmbeddingLoss()

    def _pos_list(self, x):
        return x[:, 0, :], x[:, 1, :]

    def _neg_list(self, x):
        indices_pos = []
        indices_neg = []
        x = x.reshape(-1, x.shape[2])
        for ii in range(x.shape[0] // 2 - 1):
            # for jj in range(ii + 1, x.shape[0] // 2):
            indices_pos.append(ii * 2)
            indices_pos.append(ii * 2 + 1)
            jj = random.randint(0, x.shape[0] // 2 - 1)
            while jj == ii:
                jj = random.randint(0, x.shape[0] // 2 - 1)
            indices_neg.append(jj * 2)
            indices_neg.append(jj * 2 + 1)

        indices_pos = torch.tensor(indices_pos, dtype=torch.long, device=self._get_device(x))
        indices_neg = torch.tensor(indices_neg, dtype=torch.long, device=self._get_device(x))
        return x[indices_pos], x[indices_neg]

    def _get_device(self, x):
        if x.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(x.device.type, str(x.device.index))

    def forward(self, x):
        pos1, pos2 = self._pos_list(x)
        target = torch.ones(pos1.shape[0], device=self._get_device(x))
        loss_pos = self._cs_loss(pos1, pos2, target)
        # tmp = pos1 * pos2
        # tmp = torch.mean(tmp, dim=1)
        # tmp = log1pexp(-tmp)  # torch.log(1 + torch.exp(-tmp))
        # loss_pos = tmp.mean()

        pos, neg = self._neg_list(x)
        target = -torch.ones(pos.shape[0], device=self._get_device(x))
        loss_neg = self._cs_loss(pos, neg, target)
        # tmp = pos * neg
        # tmp = torch.mean(tmp, dim=1)
        # tmp = log1pexp(tmp)  # torch.log(1 + torch.exp(-tmp))
        # loss_neg = tmp.mean()

        return loss_pos + loss_neg
