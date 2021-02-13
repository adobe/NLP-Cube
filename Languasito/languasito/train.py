import sys
import optparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
import torch

sys.path.append('')

from languasito.utils import LanguasitoDataset, load_dataset, LanguasitoCollate, Encodings
from languasito.model import Languasito


class PrintAndSaveCallback(pl.callbacks.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_validation_end(self, trainer, pl_module):
        res = pl_module._epoch_results
        if 'best_loss' in res:
            trainer.save_checkpoint('{0}.best'.format(self.args.output_base))

        trainer.save_checkpoint('{0}.last'.format(self.args.output_base))

        msg = '\n\n\tVal loss: \t{0:.4f}'.format(res['val_loss'])
        print(msg)
        print("\n")


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store', dest='train_file', default="corpus/ro-train")
    parser.add_option('--dev', action='store', dest='dev_file', default="corpus/ro-dev")
    parser.add_option('--store', action='store', dest='output_base', default="data/laro")
    parser.add_option('--resume', action='store_true', dest='resume')
    parser.add_option('--patience', action='store', default=20, type='int', dest='patience', help='Default=20')
    parser.add_option('--gpus', action='store', default=1, type='int', dest='gpus', help='Default=1')
    parser.add_option('--batch-size', action='store', default=128, type='int', dest='batch_size', help='Default=32')
    parser.add_option('--num-workers', action='store', default=4, type='int', dest='num_workers', help='Default=4')

    (params, _) = parser.parse_args(sys.argv)

    train = load_dataset(params.train_file)
    dev = load_dataset(params.dev_file)

    enc = Encodings()
    enc.update(train)
    enc.save('{0}.encodings'.format(params.output_base), full=False)

    collate = LanguasitoCollate(enc)
    model = Languasito(enc)

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

    if params.gpus == 0:
        acc = 'ddp_cpu'
    else:
        acc = 'ddp'

    # if params.resume:
    #    chk = torch.load('{0}.last'.format(params.output_base))
    # else:
    #    chk = None

    if params.resume:
        checkpoint_path = '{0}.last'.format(params.output_base)
    else:
        checkpoint_path = None

    trainer = pl.Trainer(
        gpus=params.gpus,
        accelerator=acc,
        num_nodes=1,
        default_root_dir='data/',
        callbacks=[early_stopping_callback, PrintAndSaveCallback(params)],
        val_check_interval=min(10000, len(train) // params.batch_size),
        resume_from_checkpoint=checkpoint_path,
        # limit_train_batches=5,
        # limit_val_batches=2
    )

    trainer.fit(model, train_loader, val_loader)
