import sys

sys.path.append('')
import os, argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from cube3.io_utils.objects import Document, Sentence, Token, Word
from cube3.io_utils.encodings import Encodings
from cube3.io_utils.config import TokenizerConfig
import numpy as np
from cube3.networks.modules import ConvNorm, LinearNorm
import random

from cube3.networks.utils import LMHelper, TokenizationDataset, TokenCollate

from cube3.networks.modules import WordGram


class Tokenizer(pl.LightningModule):
    def __init__(self, config: TokenizerConfig, encodings: Encodings, id2lang: {}):
        super().__init__()
        conv_layers = []
        cs_inp = 768
        NUM_FILTERS = config.cnn_filter
        for _ in range(config.cnn_layers):
            conv_layer = nn.Sequential(
                ConvNorm(cs_inp,
                         NUM_FILTERS,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(NUM_FILTERS))
            conv_layers.append(conv_layer)
            cs_inp = NUM_FILTERS // 2 + config.lang_emb_size
        self._convs = nn.ModuleList(conv_layers)
        self._lang_emb = nn.Embedding(encodings.num_langs + 1, config.lang_emb_size, padding_idx=0)
        self._output = LinearNorm(NUM_FILTERS // 2, 4)

    def forward(self, batch):
        pass

    def validation_step(self, batch):
        pass

    def validation_epoch_end(self, outputs) -> None:
        pass

    def training_step(self, batch):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


class PrintAndSaveCallback(pl.callbacks.Callback):
    def __init__(self, args, id2lang):
        super().__init__()
        self.args = args
        self._id2lang = id2lang

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # from pprint import pprint
        # pprint(metrics)
        for lang in pl_module._epoch_results:
            res = pl_module._epoch_results[lang]
            if "sent_best" in res:
                trainer.save_checkpoint(self.args.store + "." + lang + ".sent")
            if "sent_best" in res:
                trainer.save_checkpoint(self.args.store + "." + lang + ".tok")

        trainer.save_checkpoint(self.args.store + ".last")

        lang_list = [self._id2lang[ii] for ii in self._id2lang]
        s = "{0:30s}\tSENT\tTOKEN".format("Language")
        print("\n\n\t" + s)
        print("\t" + ("=" * (len(s) + 9)))
        for lang in lang_list:
            sent = metrics["val/SENT/{0}".format(lang)]
            token = metrics["val/TOKEN/{0}".format(lang)]
            msg = "\t{0:30s}:\t{1:.4f}\t{2:.4f}".format(lang, sent, token)
            print(msg)
        print("\n")


if __name__ == '__main__':
    from cube3.io_utils.misc import ArgParser

    argparser = ArgParser()
    # run argparser
    args = argparser()
    print(args)  # example

    import json

    langs = json.load(open(args.train_file))
    doc_train = Document()
    doc_dev = Document()
    id2lang = {}
    for ii in range(len(langs)):
        lang = langs[ii]
        print(lang[1], ii)
        doc_train.load(lang[1], lang_id=ii)
        doc_dev.load(lang[2], lang_id=ii)
        id2lang[ii] = lang[0]

    # ensure target dir exists
    target = args.store
    i = args.store.rfind("/")
    if i > 0:
        target = args.store[:i]
        os.makedirs(target, exist_ok=True)

    enc = Encodings()
    enc.compute(doc_train, None)
    enc.save('{0}.encodings'.format(args.store))

    config = TokenizerConfig()
    config.lm_model = args.lm_model
    if args.config_file:
        config.load(args.config_file)
        if args.lm_model is not None:
            config.lm_model = args.lm_model
    config.save('{0}.config'.format(args.store))

    # helper = LMHelper(device=args.lm_device, model=config.lm_model)
    # helper.apply(doc_dev)
    # helper.apply(doc_train)
    trainset = TokenizationDataset(doc_train)
    devset = TokenizationDataset(doc_dev)

    collate = TokenCollate(enc)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=collate.collate_fn, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(devset, batch_size=args.batch_size, collate_fn=collate.collate_fn,
                            num_workers=args.num_workers)

    model = Tokenizer(config=config, encodings=enc, id2lang=id2lang)

    # training

    early_stopping_callback = EarlyStopping(
        monitor='val/early_meta',
        patience=args.patience,
        verbose=True,
        mode='max'
    )
    if args.gpus == 0:
        acc = 'ddp_cpu'
    else:
        acc = 'ddp'
    trainer = pl.Trainer(
        gpus=args.gpus,
        accelerator=acc,
        num_nodes=1,
        default_root_dir='data/',
        callbacks=[early_stopping_callback, PrintAndSaveCallback(args, id2lang)]
        # limit_train_batches=5,
        # limit_val_batches=2,
    )

    trainer.fit(model, train_loader, val_loader)
