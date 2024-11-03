import sys
import optparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from tokenizers.trainers import WordPieceTrainer
from torch.utils.data import DataLoader
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

sys.path.append('')

from languasito.utils import LanguasitoDataset, LanguasitoCollate, LanguasitoWordGramTokenizer
from languasito.model import Languasito


class PrintAndSaveCallback(pl.callbacks.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_validation_end(self, trainer, pl_module):
        res = pl_module._epoch_results
        if 'best_loss' in res:
            pl_module.save('{0}.best'.format(self.args.output_base))

        pl_module.save('{0}.last'.format(self.args.output_base))

        msg = '\n\n\tVal loss: \t{0:.4f}'.format(res['val_loss'])
        print(msg)
        print("\n")


class CountIterator:
    def __init__(self, word_freqs):
        self._total = 0
        self._word_list = []
        self._freq_list = []
        for w in word_freqs:
            self._word_list.append(w)
            # self._freq_list.append(word_freqs[w])
            self._freq_list.append(1)
            self._total += self._freq_list[-1]

        self._w_index = 0
        self._h_index = 0

    def __len__(self):
        return self._total

    def __iter__(self):
        return self

    def __next__(self):
        if self._w_index == len(self._word_list):
            raise StopIteration
        word = self._word_list[self._w_index]
        self._h_index += 1
        if self._h_index == self._freq_list[self._w_index]:
            self._h_index = 0
            self._w_index += 1
        return word


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store', dest='train_file', default="corpus/ro-train")
    parser.add_option('--dev', action='store', dest='dev_file', default="corpus/ro-dev")
    parser.add_option('--store', action='store', dest='output_base', default="data/laro")
    parser.add_option('--resume', action='store_true', dest='resume')
    parser.add_option('--patience', action='store', default=20, type='int', dest='patience', help='Default=20')
    parser.add_option('--vocab-size', action='store', default=10000, type='int', dest='vocab_size',
                      help='Default=10000')
    parser.add_option('--gpus', action='store', default=1, type='int', dest='gpus', help='Default=1')
    parser.add_option('--batch-size', action='store', default=128, type='int', dest='batch_size', help='Default=32')
    parser.add_option('--num-workers', action='store', default=4, type='int', dest='num_workers', help='Default=4')

    (params, _) = parser.parse_args(sys.argv)

    train = LanguasitoDataset(positive_samples=4)

    train.load_file(params.train_file)

    dev = LanguasitoDataset()
    dev.load_file(params.dev_file)
    # wp = Tokenizer(WordPiece(unk_token="[UNK]"))
    wp = LanguasitoWordGramTokenizer()
    if not params.resume:
        iterator = CountIterator(train.word_freqs)
        sys.stdout.write(f'Computing wordgram... for an iterator of {len(iterator)}\n')
        sys.stdout.flush()

        wp.train_from_iterator(iterator, length=len(iterator), threshold=10)
    else:
        print(f"Loading {params.output_base}.wordpiece")
        wp.load(f'{params.output_base}.wordpiece')

    fname = f'{params.output_base}.wordpiece'
    sys.stdout.write(f'Storing {fname}... ')
    sys.stdout.flush()
    wp.save(fname)
    sys.stdout.write('done\n')
    sys.stdout.flush()

    collate = LanguasitoCollate(wp)
    model = Languasito(wp)

    train_loader = DataLoader(train, batch_size=params.batch_size, collate_fn=collate.collate_fn, shuffle=True,
                              num_workers=params.num_workers, pin_memory=True)
    val_loader = DataLoader(dev, batch_size=params.batch_size, collate_fn=collate.collate_fn,
                            num_workers=params.num_workers, pin_memory=True)

    early_stopping_callback = EarlyStopping(
        monitor='val/early_meta',
        patience=params.patience,
        verbose=True,
        mode='max'
    )

    if params.resume:
        print("resuming from previous checkpoint")
        model.load('{0}.last'.format(params.output_base))

    trainer = pl.Trainer(
        accelerator="auto",
        num_nodes=1,
        default_root_dir='data/',
        callbacks=[early_stopping_callback, PrintAndSaveCallback(params)],
        max_epochs=9999999,
        #val_check_interval=min(10000, len(train) // params.batch_size),
    )

    trainer.fit(model, train_loader, val_loader)
