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
        self._id2lang = id2lang
        self._config = config
        conv_layers = []
        cs_inp = 768 + config.lang_emb_size
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
        self._output = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, 4)

        self._dev_results = {langid: [] for langid in self._id2lang}
        self._res = {}
        for id in id2lang:
            lang = id2lang[id]
            self._res[lang] = {"sent": 0., "token": 0.}
        self._early_stop_meta_val = 0
        self._epoch_results = {}

    def forward(self, batch):
        x_emb = batch['x_input']
        x_lang = batch['x_lang']
        x_lang = self._lang_emb(x_lang).unsqueeze(1).repeat(1, x_emb.shape[1], 1)
        x = torch.cat([x_emb, x_lang], dim=-1).permute(0, 2, 1)
        x_lang = x_lang.permute(0, 2, 1)
        half = self._config.cnn_filter // 2
        res = None
        cnt = 0
        for conv in self._convs:
            conv_out = conv(x)
            tmp = torch.tanh(conv_out[:, :half, :]) * torch.sigmoid((conv_out[:, half:, :]))
            if res is None:
                res = tmp
            else:
                res = res + tmp
            x = torch.dropout(tmp, 0.2, self.training)
            cnt += 1
            if cnt != self._config.cnn_layers:
                x = torch.cat([x, x_lang], dim=1)
        x = x + res
        x = torch.cat([x, x_lang], dim=1)
        x = x.permute(0, 2, 1)
        return self._output(x)

    def validation_step(self, batch, batch_idx):
        x_lang = batch['x_lang']
        x_text = batch['x_text']
        y_offset = batch['y_offset'].cpu().numpy()
        y_target = batch['y_output'].cpu().numpy()
        y_len = batch['y_len'].cpu().numpy()
        x_l = x_lang.cpu().numpy()
        y_pred = self.forward(batch)
        y_pred = torch.argmax(y_pred, dim=-1).detach().cpu().numpy()
        for ii in range(len(y_len)):
            ofs = y_offset[ii]
            lang = x_l[ii] - 1
            for jj in range(y_len[ii]):
                self._dev_results[lang].append([x_text[ii][jj], y_target[ii, jj + ofs], y_pred[ii, jj + ofs]])

    def validation_epoch_end(self, outputs) -> None:
        # empty accumulator
        # results = {langid: {'SENT_F': 0, 'TOK_F': 0} for langid in self._id2lang}
        results = {}

        for lang in self._dev_results:
            data = self._dev_results[lang]
            g_sents = []
            p_sents = []
            tok_p = ''
            tok_g = ''
            g_sent = []
            p_sent = []
            for example in data:
                target = example[1]
                pred = example[2]
                text = example[0].replace('▁', '')
                tok_g += text
                tok_p += text
                if target == 2 or target == 3:
                    if tok_g.strip() != '':
                        g_sent.append(tok_g)
                    tok_g = ''
                if target == 3:
                    if len(g_sent) != 0:
                        g_sents.append(g_sent)
                    g_sent = []

                if pred == 2 or pred == 3:
                    if tok_p.strip():
                        p_sent.append(tok_p)
                    tok_p = ''
                if pred == 3:
                    if len(p_sent) != 0:
                        p_sents.append(p_sent)
                    p_sent = []

            if tok_g.strip() != '':
                g_sent.append(tok_g)
            if len(g_sent) != 0:
                g_sents.append(g_sent)
            if tok_p.strip() != '':
                p_sent.append(tok_p)
            if len(p_sent) != 0:
                p_sents.append(p_sent)

            sent_f, tok_f = _conll_eval(g_sents, p_sents)
            if self._id2lang is not None:
                lang = self._id2lang[lang]

            results[lang] = {}
            results[lang]['sent'] = sent_f
            results[lang]['token'] = tok_f
            self.log('val/SENT/{0}'.format(lang), sent_f)
            self.log('val/TOKEN/{0}'.format(lang), tok_f)

        self._dev_results = {langid: [] for langid in self._id2lang}
        self._epoch_results = self._compute_early_stop(results)
        self.log('val/early_meta', self._early_stop_meta_val)

    def training_step(self, batch, batch_idx):
        y_target = batch['y_output']
        y_pred = self.forward(batch)

        loss = F.cross_entropy(y_pred.view(-1, y_pred.shape[2]), y_target.view(-1), ignore_index=0)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

    def _compute_early_stop(self, res):
        for lang in res:
            if res[lang]["sent"] > self._res[lang]["sent"]:
                self._early_stop_meta_val += 1
                self._res[lang]["sent"] = res[lang]["sent"]
                res[lang]["sent_best"] = True
            if res[lang]["token"] > self._res[lang]["token"]:
                self._early_stop_meta_val += 1
                self._res[lang]["token"] = res[lang]["token"]
                res[lang]["token_best"] = True
        return res


def _conll_eval(gold, pred):
    f = open('tmp_g.txt', 'w')
    for sent in gold:
        for ii in range(len(sent)):
            head = ii
            f.write('{0}\t{1}\t_\t_\t_\t_\t{2}\t_\t_\t_\n'.format(ii + 1, sent[ii], head))
        f.write('\n')
    f.close()

    f = open('tmp_p.txt', 'w')
    for sent in pred:
        for ii in range(len(sent)):
            head = ii
            f.write('{0}\t{1}\t_\t_\t_\t_\t{2}\t_\t_\t_\n'.format(ii + 1, sent[ii], head))
        f.write('\n')
    f.close()
    from _cube.misc.conll18_ud_eval_wrapper import conll_eval
    result = conll_eval('tmp_g.txt', 'tmp_p.txt')
    if result is None:
        return 0, 0
    else:
        return result['Sentences'].f1, result['Tokens'].f1


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
            if "token_best" in res:
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
    devset = TokenizationDataset(doc_dev, shuffle=False)

    collate = TokenCollate(enc, lm_device=args.lm_device, lm_model=args.lm_model)
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
        callbacks=[early_stopping_callback, PrintAndSaveCallback(args, id2lang)],
        # limit_train_batches=5,
        # limit_val_batches=2,
    )

    trainer.fit(model, train_loader, val_loader)
