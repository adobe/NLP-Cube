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

import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append('')
from cube2.networks.modules import LinearNorm

_tok_labels = ['<PAD>', 'B', 'I', 'E', 'N', 'BS', 'ES']
_tok_label2int = {_tok_labels[ii]: ii for ii in range(len(_tok_labels))}


class Tokenizer(nn.Module):
    def __init__(self, config, encodings, num_languages=1, char_network=None):
        super(Tokenizer, self).__init__()
        self.config = config
        self.encodings = encodings
        self.char_network = char_network

        self.char_lookup = nn.Embedding(len(encodings.char2int), config.char_emb_size, padding_idx=0)
        input_emb_size = config.char_emb_size + 16
        if char_network is not None:
            input_emb_size += self.char_network._config.rnn_size * 2

        self.case_lookup = nn.Embedding(4, 16, padding_idx=0)

        config.num_languages = num_languages
        if config.num_languages != 1:
            self.lang_lookup = nn.Embedding(config.num_languages, config.lang_emb_size, padding_idx=0)
            input_emb_size += config.lang_emb_size

        self.rnn = nn.LSTM(self.config.conv_filters,
                           self.config.rnn_size,
                           self.config.rnn_layers,
                           batch_first=True,
                           bidirectional=True)

        conv_list = [nn.Conv1d(input_emb_size,
                               self.config.conv_filters,
                               self.config.conv_kernel,
                               padding=self.config.conv_kernel // 2)]
        for _ in range(self.config.conv_layers - 1):
            conv_list.append(nn.Conv1d(self.config.conv_filters,
                                       self.config.conv_filters,
                                       self.config.conv_kernel,
                                       padding=self.config.conv_kernel // 2))
            torch.nn.init.xavier_normal_(
                conv_list[-1].weight, gain=torch.nn.init.calculate_gain('tanh'))

        self.conv = nn.ModuleList(conv_list)

        self.output = nn.Sequential(LinearNorm(self.config.rnn_size * 2, 100),
                                    nn.Tanh(),
                                    nn.Dropout(0.5),
                                    LinearNorm(100, len(_tok_labels)))

        self.res = nn.Linear(self.config.conv_filters, self.config.rnn_size * 2)

    def _get_device(self):
        if self.char_lookup.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.char_lookup.weight.device.type, str(self.char_lookup.weight.device.index))

    def forward(self, chars, lang_idx=None):
        char_idx, case_idx = self._to_index(chars)
        char_idx = torch.tensor(char_idx, dtype=torch.long, device=self._get_device())
        case_idx = torch.tensor(case_idx, dtype=torch.long, device=self._get_device())
        char_emb = self.char_lookup(char_idx)
        case_emb = self.case_lookup(case_idx)

        if lang_idx is not None:
            lang_emb = self.lang_lookup(lang_idx)
            input_emb = torch.cat((char_emb, lang_emb), dim=-1)
        else:
            input_emb = char_emb

        if self.char_network is not None:
            with torch.no_grad():
                ext_char_emb, _, _ = self.char_network(chars)
                ext_char_emb = ext_char_emb.detach()
            input_emb = torch.cat((input_emb, ext_char_emb), dim=-1)

        input_emb = torch.cat((input_emb, case_emb), dim=-1)
        input_emb = input_emb.permute(0, 2, 1)
        hidden = input_emb
        res = None
        for conv in self.conv:
            c_out = conv(hidden)
            if res is not None:
                hidden = c_out + res
            else:
                hidden = c_out
            res = c_out
            hidden = torch.dropout(torch.tanh(hidden), 0.5, self.training)

        hidden = hidden.permute(0, 2, 1)

        output_rnn, hidden_rnn = self.rnn(hidden)
        output_rnn = output_rnn + self.res(hidden)
        output_rnn = torch.dropout(output_rnn, 0.5, self.training)

        output = self.output(output_rnn)
        return output

    def _to_index(self, chars):
        max_len = max([len(ex) for ex in chars])
        char_idx = np.zeros((len(chars), max_len))
        case_idx = np.zeros((len(chars), max_len))
        for ii in range(char_idx.shape[0]):
            for jj in range(char_idx.shape[1]):
                if jj < len(chars[ii]):
                    c = chars[ii][jj]
                    if c.lower() == c.upper():
                        case_id = 1
                    elif c.lower() != c:
                        case_id = 2
                    else:
                        case_id = 3
                    c = c.lower()
                    if c in self.encodings.char2int:
                        char_id = self.encodings.char2int[c]
                    else:
                        char_id = self.encodings.char2int['<UNK>']

                    char_idx[ii, jj] = char_id
                    case_idx[ii, jj] = case_id

        return char_idx, case_idx

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


def do_debug(params):
    from cube.io_utils.conll import Dataset
    trainset = Dataset()
    devset = Dataset()

    train_list = ['corpus/ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-train.conllu',
                  'corpus/ud-treebanks-v2.4/UD_French-Sequoia/fr_sequoia-ud-train.conllu',
                  'corpus/ud-treebanks-v2.4/UD_French-GSD/fr_gsd-ud-train.conllu',
                  'corpus/ud-treebanks-v2.4/UD_Portuguese-Bosque/pt_bosque-ud-train.conllu',
                  'corpus/ud-treebanks-v2.4/UD_Spanish-AnCora/es_ancora-ud-train.conllu',
                  'corpus/ud-treebanks-v2.4/UD_Catalan-AnCora/ca_ancora-ud-train.conllu',
                  'corpus/ud-treebanks-v2.4/UD_French-Spoken/fr_spoken-ud-train.conllu',
                  'corpus/ud-treebanks-v2.4/UD_Galician-CTG/gl_ctg-ud-train.conllu',
                  'corpus/ud-treebanks-v2.4/UD_Italian-ISDT/it_isdt-ud-train.conllu',
                  'corpus/ud-treebanks-v2.4/UD_Italian-PoSTWITA/it_postwita-ud-train.conllu']

    dev_list = ['corpus/ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-dev.conllu',
                'corpus/ud-treebanks-v2.4/UD_French-Sequoia/fr_sequoia-ud-dev.conllu',
                'corpus/ud-treebanks-v2.4/UD_French-GSD/fr_gsd-ud-dev.conllu',
                'corpus/ud-treebanks-v2.4/UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu',
                'corpus/ud-treebanks-v2.4/UD_Spanish-AnCora/es_ancora-ud-dev.conllu',
                'corpus/ud-treebanks-v2.4/UD_Catalan-AnCora/ca_ancora-ud-dev.conllu',
                'corpus/ud-treebanks-v2.4/UD_French-Spoken/fr_spoken-ud-dev.conllu',
                'corpus/ud-treebanks-v2.4/UD_Galician-CTG/gl_ctg-ud-dev.conllu',
                'corpus/ud-treebanks-v2.4/UD_Italian-ISDT/it_isdt-ud-dev.conllu',
                'corpus/ud-treebanks-v2.4/UD_Italian-PoSTWITA/it_postwita-ud-dev.conllu']

    # for i, t, d in zip(range(len(train_list)), train_list, dev_list):
    #     trainset.load_language(t, i)
    #     devset.load_language(d, i)
    trainset.load_language('corpus/ud-treebanks-v2.4/UD_Japanese-GSD/ja_gsd-ud-train.conllu', 0)
    devset.load_language('corpus/ud-treebanks-v2.4/UD_Japanese-GSD/ja_gsd-ud-dev.conllu', 0)
    from cube.io_utils.encodings import Encodings
    from cube2.config import CharLMConfig
    conf_char = CharLMConfig('corpus/ja-lm.conf')
    enc_char = Encodings()
    enc_char.load('corpus/ja-lm.encodings')
    from cube2.networks.charlm import CharLM
    char_lm = CharLM(conf_char, enc_char)
    char_lm.load('corpus/ja-lm.last')
    char_lm.to(params.device)
    char_lm.eval()
    # trainset.load_language('corpus/ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-train.conllu', 0)
    # devset.load_language('corpus/ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-dev.conllu', 0)

    encodings = Encodings()
    encodings.compute(trainset, devset, char_cutoff=2)

    from cube2.networks.tokenizer import Tokenizer
    from cube2.config import TokenizerConfig

    config = TokenizerConfig()

    tokenizer = Tokenizer(config, encodings, num_languages=1, char_network=char_lm)
    if params.device != 'cpu':
        tokenizer.to(params.device)

    import torch.optim as optim
    import torch.nn as nn
    trainer = optim.Adam(tokenizer.parameters(), lr=2e-3, amsgrad=True, betas=(0.9, 0.9))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if params.device != 'cpu':
        criterion.cuda(params.device)

    config.save(params.store + '.conf')
    encodings.save(params.store + '.encodings')

    _start_train(params, tokenizer, trainset, devset, criterion, trainer)


def _make_batches(dataset, batch_size=32, seq_len=1000):
    x = []
    y = []
    for example in dataset.sequences:
        lang = example[1]
        seq = example[0]
        for iWord, entry in zip(range(len(seq)), seq):
            for iChar, char in zip(range(len(entry.word)), entry.word):
                x.append(char)
                if iChar == 0:
                    y.append('B')
                elif iChar == len(entry.word) - 1:
                    y.append('E')
                else:
                    y.append('I')
                if iChar == len(entry.word) - 1 and iWord == len(seq) - 1:
                    y[-1] += 'S'
            if "spaceafter=no" not in entry.space_after.lower():
                x.append(' ')
                y.append('N')
    batches_x = []
    batches_y = []
    batch_x = []
    batch_y = []
    mbx = []
    mby = []
    for xx, yy in zip(x, y):
        batch_x.append(xx)
        batch_y.append(yy)
        if len(batch_x) == seq_len:
            mbx.append(batch_x)
            mby.append(batch_y)
            batch_x = []
            batch_y = []
        if len(mbx) == batch_size:
            batches_x.append(mbx)
            batches_y.append(mby)
            mbx = []
            mby = []
    if len(batch_x) != 0:
        mbx.append(batch_x)
        mby.append(batch_y)
    if len(mbx) != 0:
        batches_x.append(mbx)
        batches_y.append(mby)
    return batches_x, batches_y


def _compute_target(y):
    max_len = max([len(seq) for seq in y])
    y_target = np.zeros((len(y), max_len))
    for ii in range(y_target.shape[0]):
        for jj in range(y_target.shape[1]):
            if jj < len(y[ii]):
                y_target[ii, jj] = _tok_label2int[y[ii][jj]]
    return y_target


def _eval(tokenizer, dataset):
    pred_s = 0
    pred_t = 0
    ok_s = 0
    ok_t = 0
    actual_s = 0
    actual_t = 0

    batch_x, batch_y = _make_batches(dataset, batch_size=1, seq_len=-1)
    tokenizer.eval()
    target_y = batch_y[0][0]
    with torch.no_grad():
        p_y = tokenizer(batch_x[0])

    p_y = torch.argmax(p_y, dim=-1).detach().cpu().numpy()

    pred_y = []
    for index in range(p_y.shape[1]):
        pred_y.append(_tok_labels[p_y[0, index]])

    for y_target, y_pred in zip(target_y, pred_y):
        if y_target.endswith('S'):
            actual_s += 1
        if y_pred.endswith('S'):
            pred_s += 1
        if y_pred.endswith('S') and y_target.endswith('S'):
            ok_s += 1

        if y_target.startswith('B'):
            actual_t += 1
        if y_pred.startswith('B'):
            pred_t += 1
        if y_pred.startswith('B') and y_target.startswith('B'):
            ok_t += 1

    if ok_s == 0:
        f_sent = 0
    else:
        p_sent = ok_s / pred_s
        r_sent = ok_s / actual_s
        f_sent = (2 * p_sent * r_sent) / (p_sent + r_sent)
    if ok_t == 0:
        f_tok = 0
    else:
        p_tok = ok_t / pred_t
        r_tok = ok_t / actual_t
        f_tok = (2 * p_tok * r_tok) / (p_tok + r_tok)

    return f_sent, f_tok


def _start_train(params, tokenizer, trainset, devset, criterion, optimizer):
    import tqdm
    import random
    patience_left = params.patience
    f_sent, f_token = _eval(tokenizer, devset)
    print(f_sent, f_token)
    best_sent = 0
    best_tok = 0
    while patience_left > 0:
        patience_left -= 1
        random.shuffle(trainset.sequences)
        batches_x, batches_y = _make_batches(trainset)
        tokenizer.train()
        total_loss = 0
        cnt = 0
        for x, y in tqdm.tqdm(zip(batches_x, batches_y), total=len(batches_y)):
            y_pred = tokenizer(x)
            y_target = _compute_target(y)
            y_target = torch.tensor(y_target, dtype=torch.long, device=params.device)
            y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)
            y_target = y_target.reshape(y_pred.shape[0])
            loss = criterion(y_pred, y_target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
        total_loss /= cnt
        f_sent, f_token = _eval(tokenizer, devset)
        sys.stdout.write('\tTrainset avg loss = {0:.6f}\n'.format(total_loss))
        sys.stdout.write('\tDevset sent_fscore = {0:.6f} tok_fscore={1:.6f}\n'.format(f_sent, f_token))
        if f_sent > best_sent:
            best_sent = f_sent
            patience_left = params.patience
            fname = '{0}.bestSS'.format(params.store)
            sys.stdout.write('\tStoring {0}\n'.format(fname))
            tokenizer.save(fname)
        if f_token > best_tok:
            best_tok = f_token
            patience_left = params.patience
            fname = '{0}.bestTOK'.format(params.store)
            sys.stdout.write('\tStoring {0}\n'.format(fname))
            tokenizer.save(fname)
        fname = '{0}.last'.format(params.store)
        sys.stdout.write('\tStoring {0}\n'.format(fname))
        tokenizer.save(fname)


def do_test(params):
    pass


def do_tokenize(params):
    from cube.io_utils.encodings import Encodings

    encodings = Encodings()
    encodings.load(params.model_base + '.encodings')

    from cube2.networks.tokenizer import Tokenizer
    from cube2.config import TokenizerConfig

    config = TokenizerConfig()
    config.load(params.model_base + '.conf')

    tokenizer = Tokenizer(config, encodings, num_languages=1)
    tokenizer.load(params.model_base + '.last')
    if params.device != 'cpu':
        tokenizer.to(params.device)
    tokenizer.eval()

    text = open(params.test_file).read()
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    new_text = text.replace('  ', ' ')
    while new_text != text:
        text = new_text
        new_text = text.replace('  ', ' ')

    with torch.no_grad():
        x = [[ch for ch in text]]
        pred_y = tokenizer(x)
        pred_y = torch.argmax(pred_y, dim=-1)[0].detach().cpu().numpy()

    p_y = [_tok_labels[yy] for yy in pred_y]

    cw = ''
    seqs = []
    cs = []
    from cube.io_utils.conll import ConllEntry
    w_index = 1
    for index, x, y in zip(range(len(text)), text, p_y):
        if (index == len(text) - 1) or (text[index + 1] != ' '):
            spcA = 'SpaceAfter=no'
        else:
            spcA = ''

        if y.startswith('B') and cw.strip() != '':
            entry = ConllEntry(w_index, cw, '_', '_', '_', '_', 0, '_', '_', spcA)
            cs.append(entry)
            cw = ''
            w_index += 1

        if not y.startswith('N'):
            cw += x
        if y.startswith('E'):
            if cw.strip() != '':
                entry = ConllEntry(w_index, cw, '_', '_', '_', '_', 0, '_', '_', spcA)
                cs.append(entry)
                cw = ''
                w_index += 1

        if y.endswith('S'):
            if cw.strip() != '':
                entry = ConllEntry(w_index, cw, '_', '_', '_', '_', 0, '_', '_', spcA)
                cs.append(entry)

            w_index = 1
            seqs.append(cs)
            cs = []
            cw = ''

    if len(cs) > 0:
        seqs.append(cs)

    for seq in seqs:
        for entry in seq:
            sys.stdout.write(str(entry))
        print("")


if __name__ == '__main__':
    import optparse
    import sys

    parser = optparse.OptionParser()
    parser.add_option('--train', action='store_true', dest='train',
                      help='Start building a parser model')
    parser.add_option('--patience', action='store', type='int', default=20, dest='patience',
                      help='Number of epochs before early stopping (default=20)')
    parser.add_option('--store', action='store', dest='store', help='Output base', default='parser')
    parser.add_option('--batch-size', action='store', type='int', default=32, dest='batch_size',
                      help='Number of epochs before early stopping (default=32)')
    parser.add_option('--debug', action='store_true', dest='debug', help='Do some standard stuff to debug the model')
    parser.add_option('--device', action='store', dest='device', default='cpu',
                      help='What device to use for models: cpu, cuda:0, cuda:1 ...')
    parser.add_option('--test', action='store_true', dest='test', help='Test the traine model')
    parser.add_option('--test-file', action='store', dest='test_file')
    parser.add_option('--lang-id', action='store', dest='lang_id', type='int', default=0)
    parser.add_option('--model-base', action='store', dest='model_base')
    parser.add_option('--process', action='store_true')

    (params, _) = parser.parse_args(sys.argv)

    if params.debug:
        do_debug(params)
    if params.test:
        do_test(params)
    if params.process:
        do_tokenize(params)
