import optparse
import sys
import random

sys.path.append('')
import numpy as np
import torch.nn as nn
import torch.utils.data
from cube2.networks.text import TextEncoder
from cube2.config import TaggerConfig
from cube.io_utils.encodings import Encodings


class Tagger(nn.Module):
    encodings: Encodings
    config: TaggerConfig

    def __init__(self, config, encodings, num_languages=1):
        super(Tagger, self).__init__()
        self.config = config
        self.encodings = encodings
        self.num_languages = num_languages
        if num_languages == 1:
            lang_emb_size = None
        else:
            lang_emb_size = self.config.tagger_embeddings_size
            self.lang_emb = nn.Embedding(num_languages, lang_emb_size, padding_idx=0)
        self.text_network = TextEncoder(config, encodings, ext_conditioning=lang_emb_size)
        self.output_upos = nn.Linear(self.config.tagger_embeddings_size, len(self.encodings.upos2int))
        self.output_xpos = nn.Linear(self.config.tagger_embeddings_size, len(self.encodings.xpos2int))
        self.output_attrs = nn.Linear(self.config.tagger_embeddings_size, len(self.encodings.attrs2int))

    def forward(self, x):
        emb = self.text_network(x)
        s_upos = self.output_upos(emb)
        s_xpos = self.output_xpos(emb)
        s_attrs = self.output_attrs(emb)
        return s_upos, s_xpos, s_attrs


class TaggerDataset(torch.utils.data.Dataset):
    def __init__(self, conll_dataset):
        super(TaggerDataset, self).__init__()
        self.sequences = conll_dataset.sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return {'x': self.sequences[item][0]}


def _get_tgt_labels(data, encodings, device='cpu'):
    max_sent_size = 0
    for sent in data:
        if len(sent) > max_sent_size:
            max_sent_size = len(sent)
    tgt_upos = []
    tgt_xpos = []
    tgt_attrs = []
    for sent in data:
        upos_int = []
        xpos_int = []
        attrs_int = []
        for entry in sent:
            if entry.upos in encodings.upos2int:
                upos_int.append(encodings.upos2int[entry.upos])
            else:
                upos_int.append(encodings.upos2int['<UNK>'])
            if entry.xpos in encodings.xpos2int:
                xpos_int.append(encodings.xpos2int[entry.xpos])
            else:
                xpos_int.append(encodings.xpos2int['<UNK>'])
            if entry.attrs in encodings.attrs2int:
                attrs_int.append(encodings.attrs2int[entry.attrs])
            else:
                attrs_int.append(encodings.attrs2int['<UNK>'])
        for _ in range(max_sent_size - len(sent)):
            upos_int.append(encodings.upos2int['<PAD>'])
            xpos_int.append(encodings.xpos2int['<PAD>'])
            attrs_int.append(encodings.attrs2int['<PAD>'])
        tgt_upos.append(upos_int)
        tgt_xpos.append(xpos_int)
        tgt_attrs.append(attrs_int)

    import torch
    return torch.tensor(tgt_upos, device=device), torch.tensor(tgt_xpos, device=device), torch.tensor(tgt_attrs,
                                                                                                      device=device)


def _eval(tagger, dataset, encodings, device='cpu'):
    tagger.eval()
    total = 0
    upos_ok = 0
    xpos_ok = 0
    attrs_ok = 0
    num_batches = len(dataset.sequences) // params.batch_size
    if len(dataset.sequences) % params.batch_size != 0:
        num_batches += 1
    total_words = 0
    import tqdm
    pgb = tqdm.tqdm(range(num_batches), desc='\tEvaluating', ncols=80)
    tagger.eval()
    for batch_idx in pgb:
        start = batch_idx * params.batch_size
        stop = min(len(dataset.sequences), start + params.batch_size)
        data = []
        for ii in range(stop - start):
            data.append(dataset.sequences[start + ii][0])
            total_words += len(dataset.sequences[start + ii][0])
        s_upos, s_xpos, s_attrs = tagger(data)
        tgt_upos, tgt_xpos, tgt_attrs = _get_tgt_labels(data, encodings, device=device)
        s_upos = s_upos.detach().cpu().numpy()
        s_xpos = s_xpos.detach().cpu().numpy()
        s_attrs = s_attrs.detach().cpu().numpy()
        tgt_upos = tgt_upos.detach().cpu().numpy()
        tgt_xpos = tgt_xpos.detach().cpu().numpy()
        tgt_attrs = tgt_attrs.detach().cpu().numpy()
        for b_idx in range(tgt_upos.shape[0]):
            for w_idx in range(tgt_upos.shape[1]):
                pred_upos = np.argmax(s_upos[b_idx, w_idx])
                pred_xpos = np.argmax(s_xpos[b_idx, w_idx])
                pred_attrs = np.argmax(s_attrs[b_idx, w_idx])

                if tgt_upos[b_idx, w_idx] != 0:
                    total += 1
                    if pred_upos == tgt_upos[b_idx, w_idx]:
                        upos_ok += 1
                    if pred_xpos == tgt_xpos[b_idx, w_idx]:
                        xpos_ok += 1
                    if pred_attrs == tgt_attrs[b_idx, w_idx]:
                        attrs_ok += 1

    return upos_ok / total, xpos_ok / total, attrs_ok / total


def _start_train(params, trainset, devset, encodings, tagger, criterion, trainer):
    patience_left = params.patience
    epoch = 1

    best_upos = 0
    best_xpos = 0
    best_attrs = 0
    while patience_left > 0:
        sys.stdout.write('\n\nStarting epoch ' + str(epoch) + '\n')
        random.shuffle(trainset.sequences)
        num_batches = len(trainset.sequences) // params.batch_size
        if len(trainset.sequences) % params.batch_size != 0:
            num_batches += 1
        total_words = 0
        epoch_loss = 0
        import tqdm
        pgb = tqdm.tqdm(range(num_batches), desc='\tloss=NaN', ncols=80)
        tagger.train()
        for batch_idx in pgb:
            start = batch_idx * params.batch_size
            stop = min(len(trainset.sequences), start + params.batch_size)
            data = []
            for ii in range(stop - start):
                data.append(trainset.sequences[start + ii][0])
                total_words += len(trainset.sequences[start + ii][0])
            s_upos, s_xpos, s_attrs = tagger(data)
            tgt_upos, tgt_xpos, tgt_attrs = _get_tgt_labels(data, encodings, device=params.device)
            loss = criterion(s_upos.view(-1, s_upos.shape[-1]), tgt_upos.view(-1)) + criterion(
                s_xpos.view(-1, s_xpos.shape[-1]), tgt_xpos.view(-1)) + criterion(s_attrs.view(-1, s_attrs.shape[-1]),
                                                                                  tgt_attrs.view(-1))
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            epoch_loss += loss.item()
            # print("\t" + str(loss.item()))
            pgb.set_description('\tloss={0:.4f}'.format(loss.item()))

        acc_upos, acc_xpos, acc_attrs = _eval(tagger, devset, encodings)
        if best_upos < acc_upos:
            best_upos = acc_upos
            patience_left = params.patience
        if best_xpos < acc_xpos:
            best_xpos = acc_xpos
            patience_left = params.patience
        if best_attrs < acc_attrs:
            best_attrs = acc_attrs
            patience_left = params.patience
        print("\tAVG Epoch loss = {0:.6f}".format(epoch_loss / num_batches))
        print("\tValidation accuracy UPOS={0:.4f}, XPOS={1:.4f}, ATTRS={2:.4f}".format(acc_upos, acc_xpos, acc_attrs))
        epoch += 1


def do_debug(params):
    from cube.io_utils.conll import Dataset
    from cube.io_utils.encodings import Encodings
    from cube2.config import TaggerConfig

    trainset = Dataset()
    devset = Dataset()
    trainset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-train.conllu', 0)
    devset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-dev.conllu', 0)
    encodings = Encodings()
    encodings.compute(trainset, devset)
    config = TaggerConfig()
    tagger = Tagger(config, encodings, 1)
    if params.device != 'cpu':
        tagger.cuda(params.device)

    import torch.optim as optim
    import torch.nn as nn
    trainer = optim.Adam(tagger.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if params.device != 'cpu':
        criterion.cuda(params.device)
    _start_train(params, trainset, devset, encodings, tagger, criterion, trainer)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store_true', dest='train',
                      help='Start building a tagger model')
    parser.add_option('--patience', action='store', type='int', default=20, dest='patience',
                      help='Number of epochs before early stopping (default=20)')
    parser.add_option('--batch-size', action='store', type='int', default=32, dest='batch_size',
                      help='Number of epochs before early stopping (default=32)')
    parser.add_option('--debug', action='store_true', dest='debug', help='Do some standard stuff to debug the model')
    parser.add_option('--device', action='store', dest='device', default='cpu',
                      help='What device to use for models: cpu, cuda:0, cuda:1 ...')

    (params, _) = parser.parse_args(sys.argv)

    if params.debug:
        do_debug(params)
