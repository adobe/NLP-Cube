import torch
import torch.nn as nn
import numpy as np
import optparse
import sys
import tqdm

sys.path.append('')


class CharLM(nn.Module):
    def __init__(self, config, encodings):
        super(CharLM, self).__init__()
        self._config = config
        self._encodings = encodings
        self._char_emb = nn.Embedding(len(encodings.char2int), config.char_emb_size)
        self._case_emb = nn.Embedding(4, 16)
        self._forward_rnn = nn.LSTM(config.char_emb_size + 16, config.rnn_size, config.rnn_layers, batch_first=True)
        self._backward_rnn = nn.LSTM(config.char_emb_size + 16, config.rnn_size, config.rnn_layers, batch_first=True)
        self._output_softmax = nn.Linear(config.rnn_size, len(encodings.char2int))

    def forward(self, sequences):
        input = sequences
        m = max([len(seq) for seq in input])
        x_fw_char = np.zeros((len(input), m + 2))
        x_bw_char = np.zeros((len(input), m + 2))
        x_fw_case = np.zeros((len(input), m + 2))
        x_bw_case = np.zeros((len(input), m + 2))
        for iBatch in range(len(input)):
            for iChar in range(m):
                if iChar < len(input[iBatch]):
                    char = input[iBatch][iChar]
                    if char.lower() == char.upper():
                        case_idx = 1
                    elif char.lower() != char:
                        case_idx = 2
                    else:
                        case_idx = 3
                    char = char.lower()
                    if char in self._encodings.char2int:
                        char_idx = self._encodings.char2int[char]
                    else:
                        char_idx = self._encodings.char2int['<UNK>']
                    x_fw_char[iBatch, iChar + 1] = char_idx
                    x_fw_case[iBatch, iChar + 1] = case_idx
                    x_bw_char[iBatch, x_bw_char.shape[1] - iChar - 2] = char_idx
                    x_bw_case[iBatch, x_bw_case.shape[1] - iChar - 2] = case_idx

        x_fw_char = torch.tensor(x_fw_char, dtype=torch.long, device=self._get_device())
        x_bw_char = torch.tensor(x_bw_char, dtype=torch.long, device=self._get_device())
        x_fw_case = torch.tensor(x_fw_case, dtype=torch.long, device=self._get_device())
        x_bw_case = torch.tensor(x_bw_case, dtype=torch.long, device=self._get_device())

        x_fw_char = self._char_emb(x_fw_char)
        x_bw_char = self._char_emb(x_bw_char)
        x_fw_case = self._case_emb(x_fw_case)
        x_bw_case = self._case_emb(x_bw_case)

        x_fw = torch.cat((x_fw_char, x_fw_case), dim=-1)
        x_bw = torch.cat((x_bw_char, x_bw_case), dim=-1)
        hidden_fw = self._forward_rnn(x_fw)[0]
        hidden_bw = self._reverse(self._backward_rnn(x_bw)[0])
        output_fw = self._output_softmax(hidden_fw)
        output_bw = self._output_softmax(hidden_bw)

        return torch.cat((hidden_fw, hidden_bw), dim=-1)[:, 1:-1, :], output_fw, output_bw

    def _get_device(self):
        if self._case_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._case_emb.weight.device.type, str(self._case_emb.weight.device.index))

    def _reverse(self, x):
        x_list = []
        for ii in range(x.shape[1]):
            x_list.append(x[:, x.shape[1] - ii - 1, :].unsqueeze(1))

        return torch.cat(x_list, dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


class DataLoader:
    def __init__(self, filename):
        self._text = open(filename).read().replace('\n', ' ').replace('\r', ' ')
        self._index = 0

    def read_next(self, batch_size=16, sequence_size=500):
        to_read = sequence_size * batch_size
        stop = self._index + to_read
        m = min([len(self._text), stop])
        left = stop - len(self._text)
        seq = self._text[self._index:m]
        if left <= 0:
            self._index += to_read
        else:
            self._index = left
            for ii in range(left):
                seq.append(self._text[ii])
        batch = []
        for ii in range(batch_size):
            batch.append(seq[ii * sequence_size:(ii + 1) * sequence_size])
        return batch


def _update_encodings(encodings, filename, cutoff=2):
    lines = open(filename).readlines()
    char2count = {}
    for line in tqdm.tqdm(lines):
        line = line.replace('\n', '').replace('\r', '')
        for char in line:
            char = char.lower()
            if char in char2count:
                char2count[char] += 1
            else:
                char2count[char] = 1

    for char in char2count:
        if char2count[char] > cutoff:
            if char not in encodings.char2int:
                encodings.char2int[char] = len(encodings.char2int)
                encodings.characters.append(char)


def _get_target(x, encodings):
    target = np.zeros((len(x), len(x[0])))
    for iBatch in range(target.shape[0]):
        for iChar in range(target.shape[1]):
            char = x[iBatch][iChar].lower()
            if char in encodings.char2int:
                target[iBatch, iChar] = encodings.char2int[char]
            else:
                target[iBatch, iChar] = encodings.char2int['<UNK>']
    return target


def _start_train(params):
    from cube2.config import CharLMConfig
    from cube.io_utils.encodings import Encodings
    if params.config_file:
        config = CharLMConfig(filename=params.config_file)
    else:
        config = CharLMConfig()

    config.save('{0}.conf'.format(params.store))
    encodings = Encodings()
    encodings.char2int['<PAD>'] = 0
    encodings.char2int['<UNK>'] = 1
    encodings.char2int[' '] = 2
    encodings.characters.append('<PAD>')
    encodings.characters.append('<UNK>')
    encodings.characters.append(' ')
    _update_encodings(encodings, params.train_file)

    encodings.save('{0}.encodings'.format(params.store))
    sys.stdout.write('Found {0} unique characters after cutoff\n'.format(len(encodings.char2int)))
    patience_left = params.patience
    model = CharLM(config, encodings)
    model.to(params.device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    best_loss = 9999
    global_step = 0
    train = DataLoader(params.train_file)
    dev = DataLoader(params.dev_file)
    while patience_left > 0:
        model.train()
        patience_left -= 1
        total_loss = 0
        pgb = tqdm.tqdm(range(500))
        pgb.set_description('loss=N/A    ')
        for step in pgb:
            x = train.read_next(params.batch_size)
            emb, y_fw, y_bw = model(x)
            y_tar = _get_target(x, encodings)
            y_tar = torch.tensor(y_tar, dtype=torch.long, device=params.device)

            y_fw = y_fw[:, :-2, :]
            y_bw = y_bw[:, 2:, :]
            y_fw = y_fw.reshape(-1, y_fw.shape[-1])
            y_bw = y_bw.reshape(-1, y_bw.shape[-1])
            y_tar = y_tar.view(-1)
            loss = loss_func(y_fw, y_tar) + \
                   loss_func(y_bw, y_tar)

            total_loss += loss.item()
            pgb.set_description('loss={0}'.format(total_loss / (step + 1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss /= 500
        global_step += 500
        sys.stdout.write('\tAVG loss for global step {0} is {1}\n'.format(global_step, total_loss))
        fn = '{0}.last'.format(params.store)
        sys.stdout.write('\tStoring {0}\n'.format(fn))
        model.save(fn)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store_true', dest='train',
                      help='Start building a tagger model')
    parser.add_option('--config', action='store', dest='config_file',
                      help='Use this configuration file for tagger')
    parser.add_option('--patience', action='store', type='int', default=20, dest='patience',
                      help='Number of epochs before early stopping (default=20)')
    parser.add_option('--store', action='store', dest='store', help='Output base', default='tagger')
    parser.add_option('--batch-size', action='store', type='int', default=32, dest='batch_size',
                      help='Number of epochs before early stopping (default=32)')
    parser.add_option('--debug', action='store_true', dest='debug', help='Do some standard stuff to debug the model')
    parser.add_option('--device', action='store', dest='device', default='cpu',
                      help='What device to use for models: cpu, cuda:0, cuda:1 ...')
    parser.add_option('--test', action='store_true', dest='test', help='Test the traine model')
    parser.add_option('--test-file', action='store', dest='test_file')
    parser.add_option('--lang-id', action='store', dest='lang_id', type='int', default=0)
    parser.add_option('--model-base', action='store', dest='model_base')
    parser.add_option('--train-file', action='store', dest='train_file')
    parser.add_option('--dev-file', action='store', dest='dev_file')

    (params, _) = parser.parse_args(sys.argv)

    if params.train:
        _start_train(params)
