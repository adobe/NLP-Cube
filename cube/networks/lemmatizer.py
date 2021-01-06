import sys
sys.path.append('')

import optparse
import torch
import tqdm
import torch.nn as nn
import numpy as np
from cube.io_utils.encodings import Encodings
from cube.io_utils.config import LemmatizerConfig
from cube.networks.modules import LinearNorm, ConvNorm, Attention


class Lemmatizer(nn.Module):
    encodings: Encodings
    config: LemmatizerConfig

    def __init__(self, config, encodings, num_languages=1, target_device='cpu'):
        super(Lemmatizer, self).__init__()
        self._config = config
        self._encodings = encodings
        self._num_languages = num_languages
        self._target_device = target_device

        self._char_list = ['' for char in encodings.char2int]
        for char in encodings.char2int:
            self._char_list[encodings.char2int[char]] = char
        self._lang_emb = nn.Embedding(num_languages + 1, config.lang_emb_size, padding_idx=0)
        self._upos_emb = nn.Embedding(len(encodings.upos2int), config.upos_emb_size, padding_idx=0)
        self._char_emb = nn.Embedding(len(encodings.char2int) + 2, config.char_emb_size,
                                      padding_idx=0)  # start/stop index
        self._case_emb = nn.Embedding(5, 16, padding_idx=0)  # 0-pad 1-upper 2-lower 3-symbol 4-number
        self._start_index = len(encodings.char2int)
        self._stop_index = len(encodings.char2int) + 1
        convolutions = []
        cs_inp = config.char_emb_size + config.lang_emb_size + config.upos_emb_size + 16
        for _ in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(cs_inp,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
            cs_inp = 512
        self._char_conv = nn.ModuleList(convolutions)
        encoder_layers = []
        for ii in range(config.encoder_layers):
            encoder_layers.append(nn.LSTM(cs_inp, config.encoder_size, 1, batch_first=True, bidirectional=True))
            cs_inp = config.encoder_size * 2 + config.lang_emb_size + config.upos_emb_size + 16

        self._encoder_layers = nn.ModuleList(encoder_layers)
        self._decoder = nn.LSTM(cs_inp + config.char_emb_size, config.decoder_size, config.decoder_layers,
                                batch_first=True, bidirectional=False)
        self._attention = Attention(cs_inp // 2, config.decoder_size, config.att_proj_size)

        self._output_char = LinearNorm(config.decoder_size, len(self._encodings.char2int))  # we use <PAD> for stop (0)
        self._output_case = LinearNorm(config.decoder_size, 3)  # we use <PAD> for stop (0)
        self._start_frame = nn.Embedding(1,
                                         config.encoder_size * 2 + config.char_emb_size + config.lang_emb_size + config.upos_emb_size + 16)

    def forward(self, x_chars, x_lang, x_upos, gs_output=None):
        char_idx, case_idx, upos_idx, lang_idx = self._make_batch(x_chars, x_upos, x_lang)
        char_emb = self._char_emb(char_idx)
        case_emb = self._case_emb(case_idx)
        upos_emb = self._upos_emb(upos_idx)
        lang_emb = self._lang_emb(lang_idx)
        conditioning = torch.cat((case_emb, upos_emb, lang_emb), dim=-1)
        if gs_output is not None:
            output_idx = self._get_output_idx(gs_output)

        x = torch.cat((char_emb, conditioning), dim=-1)
        x = x.permute(0, 2, 1)
        for conv in self._char_conv:
            x = torch.dropout(torch.relu(conv(x)), 0.5, self.training)

        x = x.permute(0, 2, 1)
        output = x
        for ii in range(self._config.encoder_layers):
            output, _ = self._encoder_layers[ii](output)
            tmp = torch.cat((output, conditioning), dim=-1)
            output = tmp

        encoder_output = output

        step = 0
        done = np.zeros(encoder_output.shape[0])
        start_frame = self._start_frame(
            torch.tensor([0], dtype=torch.long, device=self._get_device())).unsqueeze(1).repeat(encoder_output.shape[0],
                                                                                                1, 1)
        decoder_output, decoder_hidden = self._decoder(start_frame)

        out_char_list = []
        out_case_list = []
        while True:
            if gs_output is not None:
                if step == output_idx.shape[1]:
                    break
            elif np.sum(done) == encoder_output.shape[0]:
                break
            elif step == encoder_output.shape[1] * 20:  # failsafe
                break

            att = self._attention(decoder_hidden[-1][-1, :, :], encoder_output)
            context = torch.bmm(att.unsqueeze(1), encoder_output)

            if step == 0:
                prev_char_emb = torch.zeros((encoder_output.shape[0], 1, self._config.char_emb_size),
                                            device=self._get_device())

            decoder_input = torch.cat((context, prev_char_emb), dim=-1)
            decoder_output, decoder_hidden = self._decoder(decoder_input,
                                                           hx=(torch.dropout(decoder_hidden[0], 0.5, self.training),
                                                               torch.dropout(decoder_hidden[1], 0.5, self.training)))

            output_char = self._output_char(decoder_output)
            output_case = self._output_case(decoder_output)
            out_char_list.append(output_char.unsqueeze(1))
            out_case_list.append(output_case.unsqueeze(1))
            selected_chars = torch.argmax(output_char, dim=-1)
            for ii in range(selected_chars.shape[0]):
                if selected_chars[ii].squeeze() == 0:
                    done[ii] = 1
            if gs_output is not None:
                prev_char_emb = self._char_emb(output_idx[:, step]).unsqueeze(1)
            else:
                prev_char_emb = self._char_emb(selected_chars)
            step += 1

        return torch.cat(out_char_list, dim=1), torch.cat(out_case_list, dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self._target_device))

    def _get_device(self):
        if self._char_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._char_emb.weight.device.type, str(self._char_emb.weight.device.index))

    def _make_batch(self, x_chars, x_uposes, x_langs):
        max_len = max([len(x) for x in x_chars])
        x_char = np.zeros((len(x_chars), max_len + 2))
        x_case = np.zeros((x_char.shape[0], x_char.shape[1]))
        x_upos = np.zeros((x_char.shape[0], x_char.shape[1]))
        x_lang = np.zeros((x_char.shape[0], x_char.shape[1]))
        for ii in range(x_char.shape[0]):
            for jj in range(x_char.shape[1]):
                if jj == 0:
                    x_char[ii, jj] = self._start_index
                elif jj == len(x_chars[ii]) + 1:
                    x_char[ii, jj] = self._stop_index
                elif jj < len(x_chars[ii]) + 1:
                    char = x_chars[ii][jj - 1]
                    case = _get_case(char)
                    lang = x_langs[ii] + 1
                    upos = x_uposes[ii]
                    if char.lower() in self._encodings.char2int:
                        char = self._encodings.char2int[char.lower()]
                    else:
                        char = self._encodings.char2int['<UNK>']
                    x_char[ii, jj] = char
                    x_case[ii, jj] = case
                    x_upos[ii, jj] = self._encodings.upos2int[upos]
                    x_lang[ii, jj] = lang
        x_char = torch.tensor(x_char, dtype=torch.long, device=self._get_device())
        x_case = torch.tensor(x_case, dtype=torch.long, device=self._get_device())
        x_upos = torch.tensor(x_upos, dtype=torch.long, device=self._get_device())
        x_lang = torch.tensor(x_lang, dtype=torch.long, device=self._get_device())
        return x_char, x_case, x_upos, x_lang

    def _get_output_idx(self, x):
        tmp = [[0, 0, 0, lemma] for lemma in x]
        target_char, target_case = _get_targets(tmp, self._encodings)
        return torch.tensor(target_char, dtype=torch.long, device=self._get_device())

    def process (self, sequences, lang_id):
        x_chars, x_upos, x_lang = [], [], []

        for seq in sequences:
            for elem in seq:
                x_chars.append(elem.word)
                x_upos.append(elem.upos)
                x_lang.append(lang_id)

        y_char_pred, y_case_pred = self.forward(x_chars, x_lang, x_upos)
        y_char_pred = torch.argmax(y_char_pred, dim=-1).cpu().numpy()
        y_case_pred = torch.argmax(y_case_pred, dim=-1).cpu().numpy()

        lemmas = []
        for ii in range(y_char_pred.shape[0]):
            lemma_pred = ''
            for jj in range(y_char_pred.shape[1]):
                if y_char_pred[ii, jj] == 0:
                    break
                else:
                    char = self._char_list[y_char_pred[ii, jj].item()]
                    if y_case_pred[ii, jj] == 1:
                        char = char.upper()
                    lemma_pred += char
            lemmas.append(lemma_pred)

        cnt = 0
        for seq in sequences:
            for elem in seq:
                elem.lemma = lemmas[cnt]
                cnt+=1

        return sequences


def _eval(lemmatizer, dataset):
    ok = 0
    total = 0
    batches = _get_batches(dataset, 32)
    lemmatizer.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(batches, ncols=80, desc='\tEvaluating'):
            x_chars = []
            x_lang = []
            x_upos = []
            for ii in range(len(batch)):
                x_chars.append(batch[ii][0])
                x_lang.append(batch[ii][1])
                x_upos.append(batch[ii][2])

            y_char_target, y_case_target = _get_targets(batch, lemmatizer._encodings)
            y_char_pred, y_case_pred = lemmatizer(x_chars, x_upos, x_lang)
            good = np.ones(y_char_target.shape[0])
            y_char_pred = torch.argmax(y_char_pred, dim=-1).cpu().numpy()
            y_case_pred = torch.argmax(y_case_pred, dim=-1).cpu().numpy()
            for ii in range(y_char_target.shape[0]):
                lemma_target = batch[ii][3]
                lemma_pred = ''
                for jj in range(y_char_pred.shape[1]):
                    if y_char_pred[ii, jj] == 0:
                        break
                    else:
                        char = lemmatizer._char_list[y_char_pred[ii, jj].item()]
                        if y_case_pred[ii, jj] == 1:
                            char = char.upper()
                        lemma_pred += char
                if lemma_target == lemma_pred:
                    good[ii] = 1
                else:
                    good[ii] = 0

            total += y_char_target.shape[0]
            ok += np.sum(good)
            sys.stderr.flush()

    return ok / total


def _extract_data(dataset, unique=True):
    data = []
    visited = {}
    for example in dataset.sequences:
        seq = example[0]
        lang_id = example[1]
        for entry in seq:
            key = '{0}_{1}_{2}'.format(lang_id, entry.upos, entry.word)
            if not unique or key not in visited:
                visited[key] = 1
                if len(entry.word) < 100 and len(entry.lemma) < 100:
                    data.append([entry.word, entry.upos, lang_id, entry.lemma])
                else:
                    print(entry.word, lang_id)

    return data


def _get_batches(dataset, batch_size):
    batches = []
    batch = []
    for ii in range(len(dataset)):
        batch.append(dataset[ii])
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
    if len(batch) != 0:
        batches.append(batch)
    return batches


def _get_case(char):
    numbers = {'{0}'.format(ii): 1 for ii in range(10)}
    if char in numbers:
        return 1
    elif char.lower() == char.upper():
        return 2
    elif char.lower() != char:
        return 3
    else:
        return 4


def _get_targets(batch, encodings):
    lemmas = [example[3] for example in batch]
    max_len = max([len(lemma) for lemma in lemmas]) + 1
    targets_char = np.zeros((len(lemmas), max_len))
    targets_case = np.zeros((len(lemmas), max_len))
    for ii in range(targets_char.shape[0]):
        for jj in range(targets_char.shape[1]):
            if jj < len(lemmas[ii]):
                char = lemmas[ii][jj]
                if char.lower() != char:
                    case = 1
                else:
                    case = 2
                targets_case[ii, jj] = case
                if char.lower() in encodings.char2int:
                    targets_char[ii, jj] = encodings.char2int[char.lower()]
                else:
                    targets_char[ii, jj] = encodings.char2int['<UNK>']

    return targets_char, targets_case


def _start_train(lemmatizer, trainset, devset, params):
    import random
    patience_left = params.patience
    trainer = torch.optim.Adam(lemmatizer.parameters(), lr=1e-4)
    critetion = torch.nn.CrossEntropyLoss()
    # prepare dataset
    trainset = _extract_data(trainset, unique=True)
    devset = _extract_data(devset, unique=True)
    epoch = 0
    best_score = 0  # _eval(lemmatizer, devset)

    lemmatizer._config.save('{0}.conf'.format(params.store))
    lemmatizer._encodings.save('{0}.encodings'.format(params.store))
    sys.stdout.write('Devset accuracy {0}\n'.format(best_score))
    sys.stdout.flush()

    while patience_left > 0:
        patience_left -= 1
        epoch += 1
        sys.stdout.write('Epoch {0}\n'.format(epoch))
        sys.stdout.flush()
        sys.stdout.write('\tShuffling training data\n')
        sys.stdout.flush()
        random.shuffle(trainset)
        batches = _get_batches(trainset, params.batch_size)
        pgb = tqdm.tqdm(batches, desc='\tloss=NaN', ncols=80)
        total_loss = 0
        lemmatizer.train()
        for batch in pgb:
            x_chars = []
            x_lang = []
            x_upos = []
            x_lemmas = []
            for ii in range(len(batch)):
                x_chars.append(batch[ii][0])
                x_lang.append(batch[ii][1])
                x_upos.append(batch[ii][2])
                x_lemmas.append(batch[ii][3])

            y_char_target, y_case_target = _get_targets(batch, encodings=lemmatizer._encodings)
            y_char_pred, y_case_pred = lemmatizer(x_chars, x_upos, x_lang, gs_output=x_lemmas)
            y_char_target = torch.tensor(y_char_target, dtype=torch.long, device=lemmatizer._get_device()).view(-1)
            y_case_target = torch.tensor(y_case_target, dtype=torch.long, device=lemmatizer._get_device()).view(-1)
            y_char_pred = y_char_pred.view(-1, y_char_pred.shape[-1])
            y_case_pred = y_case_pred.view(-1, y_case_pred.shape[-1])
            loss = critetion(y_char_pred, y_char_target) + critetion(y_case_pred, y_case_target)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            loss = loss.item()
            total_loss += loss
            pgb.set_description('\tloss={0:.4f}'.format(loss))
            sys.stderr.flush()

        acc = _eval(lemmatizer, devset)
        sys.stderr.flush()
        sys.stdout.write('\t\tAVG epoch loss {0}\n'.format(total_loss / len(batches)))
        sys.stdout.write('\t\tDevset accuracy {0}\n'.format(acc))
        sys.stdout.flush()
        if acc > best_score:
            best_score = acc
            fname = '{0}.best'.format(params.store)
            sys.stdout.write('\t\tStoring {0}\n'.format(fname))
            lemmatizer.save(fname)
            patience_left = params.patience

        fname = '{0}.last'.format(params.store)
        sys.stdout.write('\t\tStoring {0}\n'.format(fname))
        lemmatizer.save(fname)


def do_train(params):
    import json
    ds_list = json.load(open(params.train_file))
    train_list = []
    dev_list = []
    for ii in range(len(ds_list)):
        train_list.append(ds_list[ii][1])
        dev_list.append(ds_list[ii][2])

    from cube.io_utils.conll import Dataset
    from cube.io_utils.encodings import Encodings

    trainset = Dataset()
    devset = Dataset()
    for ii, train, dev in zip(range(len(train_list)), train_list, dev_list):
        trainset.load_language(train, ii, ignore_compound=True)
        devset.load_language(dev, ii, ignore_compound=True)
    encodings = Encodings()
    if params.resume:
        encodings.load('{0}.encodings'.format(params.store))
    else:
        encodings.compute(trainset, devset, word_cutoff=2)

    config = LemmatizerConfig(filename=params.config_file)
    config.num_languages = len(train_list)
    lemmatizer = Lemmatizer(config, encodings, num_languages=len(train_list))
    if params.resume:
        lemmatizer.load('{0}.last'.format(params.store))
    lemmatizer.to(params.device)
    _start_train(lemmatizer, trainset, devset, params)


def do_test():
    pass


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store', dest='train_file',
                      help='Start building a tagger model')
    parser.add_option('--config', action='store', dest='config_file',
                      help='Use this configuration file for tagger')
    parser.add_option('--patience', action='store', type='int', default=20, dest='patience',
                      help='Number of epochs before early stopping (default=20)')
    parser.add_option('--store', action='store', dest='store', help='Output base', default='tagger')
    parser.add_option('--batch-size', action='store', type='int', default=32, dest='batch_size',
                      help='Number of epochs before early stopping (default=32)')
    parser.add_option('--device', action='store', dest='device', default='cpu',
                      help='What device to use for models: cpu, cuda:0, cuda:1 ...')
    parser.add_option('--test', action='store_true', dest='test', help='Test the traine model')
    parser.add_option('--test-file', action='store', dest='test_file')
    parser.add_option('--lang-id', action='store', dest='lang_id', type='int', default=0)
    parser.add_option('--model-base', action='store', dest='model_base')
    parser.add_option('--resume', action='store_true', dest='resume')

    (params, _) = parser.parse_args(sys.argv)

    if params.train_file:
        do_train(params)
    elif params.test:
        do_test(params)
