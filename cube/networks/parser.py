import sys

from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

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
from cube.io_utils.objects import Document, Sentence, Token, Word
from cube.io_utils.encodings import Encodings
from cube.io_utils.config import ParserConfig
import numpy as np
from cube.networks.modules import ConvNorm, LinearNorm, BilinearAttention, Attention, MLP, DeepBiaffine
import random

from cube.networks.utils import MorphoCollate, MorphoDataset, GreedyDecoder, ChuLiuEdmondsDecoder, unpack, mask_concat

from cube.networks.modules import WordGram


class Parser(pl.LightningModule):
    def __init__(self, config: ParserConfig, encodings: Encodings, language_codes: [] = None, ext_word_emb=0):
        super().__init__()
        self._config = config
        self._encodings = encodings
        if not isinstance(ext_word_emb, list):
            ext_word_emb = [ext_word_emb]
        self._ext_word_emb = ext_word_emb

        self._word_net = WordGram(len(encodings.char2int), num_langs=encodings.num_langs + 1,
                                  num_filters=config.char_filter_size, char_emb_size=config.char_emb_size,
                                  lang_emb_size=config.lang_emb_size, num_layers=config.char_layers)
        self._zero_emb = nn.Embedding(1, config.char_filter_size // 2)
        self._num_langs = encodings.num_langs
        self._language_codes = language_codes

        ext2int = []
        for input_size in self._ext_word_emb:
            module = MLP(input_size, config.external_proj_size)
            ext2int.append(module)
        self._ext_proj = nn.ModuleList(ext2int)

        conv_layers = []
        cs_inp = config.char_filter_size // 2 + config.lang_emb_size + config.word_emb_size + config.external_proj_size
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
        self._word_emb = nn.Embedding(len(encodings.word2int), config.word_emb_size, padding_idx=0)
        self._lang_emb = nn.Embedding(encodings.num_langs + 1, config.lang_emb_size, padding_idx=0)
        self._convs = nn.ModuleList(conv_layers)

        self._aupos = LinearNorm(config.char_filter_size // 2 + config.lang_emb_size, len(encodings.upos2int))
        self._axpos = LinearNorm(config.char_filter_size // 2 + config.lang_emb_size, len(encodings.xpos2int))
        self._aattrs = LinearNorm(config.char_filter_size // 2 + config.lang_emb_size, len(encodings.attrs2int))

        self._pre_morpho = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, NUM_FILTERS // 2)
        self._upos = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.upos2int))
        self._attrs = LinearNorm(64 + NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.attrs2int))
        self._xpos = LinearNorm(64 + NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.xpos2int))
        self._upos_emb = nn.Embedding(len(encodings.upos2int), 64)

        self._rnn = nn.LSTM(NUM_FILTERS // 2 + config.lang_emb_size + config.external_proj_size, config.rnn_size,
                            num_layers=config.rnn_layers, batch_first=True, bidirectional=True, dropout=0.33)

        self._pre_out = LinearNorm(config.rnn_size * 2 + config.lang_emb_size, config.pre_parser_size)
        # self._head_r1 = LinearNorm(config.pre_parser_size, config.head_size)
        # self._head_r2 = LinearNorm(config.pre_parser_size, config.head_size)
        # self._label_r1 = LinearNorm(config.pre_parser_size, config.label_size)
        # self._label_r2 = LinearNorm(config.pre_parser_size, config.label_size)
        # self._att_net = BilinearAttention(config.head_size, config.head_size)
        # self._label_linear = nn.Linear(config.label_size * 2, len(encodings.label2int))
        # self._label_bilinear = nn.Bilinear(config.label_size, config.label_size, len(encodings.label2int))
        self._head = DeepBiaffine(config.pre_parser_size, config.pre_parser_size, config.head_size, 1, dropout=0.1)
        self._label = DeepBiaffine(config.pre_parser_size, config.pre_parser_size, config.label_size,
                                   len(encodings.label2int), dropout=0.1, pairwise=False)
        self._r_emb = nn.Embedding(1,
                                   config.char_filter_size // 2 + config.lang_emb_size + config.word_emb_size + config.external_proj_size)

        # self._decoder = GreedyDecoder()
        self._decoder = ChuLiuEdmondsDecoder()

        if self._language_codes:
            self._res = {}
            for language_code in self._language_codes:
                self._res[language_code] = {"upos": 0., "attrs": 0., 'uas': 0., 'las': 0.}
            self._early_stop_meta_val = 0

    def _compute_early_stop(self, res):
        for lang in res:
            if res[lang]["uas"] > self._res[lang]["uas"]:
                self._early_stop_meta_val += 1
                self._res[lang]["uas"] = res[lang]["uas"]
                res[lang]["uas_best"] = True
            if res[lang]["las"] > self._res[lang]["las"]:
                self._early_stop_meta_val += 1
                self._res[lang]["las"] = res[lang]["las"]
                res[lang]["las_best"] = True
        return res

    def forward(self, X):
        x_sents = X['x_sent']
        x_lang_sent = X['x_lang_sent']
        x_words_chars = X['x_word']
        x_words_case = X['x_word_case']
        x_lang_word = X['x_lang_word']
        x_sent_len = X['x_sent_len']
        x_word_len = X['x_word_len']
        x_sent_masks = X['x_sent_masks']
        x_word_masks = X['x_word_masks']
        x_word_emb_packed = X['x_word_embeddings']
        gs_upos = None
        if 'y_upos' in X:
            gs_upos = X['y_upos']
        char_emb_packed = self._word_net(x_words_chars, x_words_case, x_lang_word, x_word_masks, x_word_len)

        sl = x_sent_len.cpu().numpy()

        char_emb = unpack(char_emb_packed, sl, x_sents.shape[1], device=self._get_device())
        word_emb_ext = None

        for ii in range(len(x_word_emb_packed)):
            we = unpack(x_word_emb_packed[ii], sl, x_sents.shape[1], self._get_device())
            if word_emb_ext is None:
                word_emb_ext = self._ext_proj[ii](we)
            else:
                word_emb_ext = word_emb_ext + self._ext_proj[ii](we)

        word_emb_ext = word_emb_ext / len(x_word_emb_packed)
        word_emb_ext = torch.tanh(word_emb_ext)

        lang_emb = self._lang_emb(x_lang_sent)
        lang_emb = lang_emb.unsqueeze(1).repeat(1, char_emb.shape[1] + 1, 1)

        aupos = self._aupos(torch.cat([char_emb, lang_emb[:, 1:, :]], dim=-1))
        aattrs = self._aattrs(torch.cat([char_emb, lang_emb[:, 1:, :]], dim=-1))
        axpos = self._axpos(torch.cat([char_emb, lang_emb[:, 1:, :]], dim=-1))

        word_emb = self._word_emb(x_sents)

        x = mask_concat([word_emb, char_emb, word_emb_ext], 0.33, self.training, self._get_device())

        x = torch.cat([x, lang_emb[:, 1:, :]], dim=-1)
        # prepend root
        root_emb = self._r_emb(torch.zeros((x.shape[0], 1), device=self._get_device(), dtype=torch.long))
        x = torch.cat([root_emb, x], dim=1)
        x = x.permute(0, 2, 1)
        lang_emb = lang_emb.permute(0, 2, 1)
        half = self._config.cnn_filter // 2
        res = None
        hidden = None
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
            if cnt == self._config.aux_softmax_location:
                hidden = torch.cat([x + res, lang_emb], dim=1)
            if cnt != self._config.cnn_layers:
                x = torch.cat([x, lang_emb], dim=1)

        x = x + res
        x_parse = x.permute(0, 2, 1)
        # aux tagging
        lang_emb = lang_emb.permute(0, 2, 1)
        hidden = hidden.permute(0, 2, 1)[:, 1:, :]
        pre_morpho = torch.dropout(torch.tanh(self._pre_morpho(hidden)), 0.33, self.training)
        pre_morpho = torch.cat([pre_morpho, lang_emb[:, 1:, :]], dim=2)
        upos = self._upos(pre_morpho)
        if gs_upos is None:
            upos_idx = torch.argmax(upos, dim=-1)
        else:
            upos_idx = gs_upos

        upos_emb = self._upos_emb(upos_idx)
        attrs = self._attrs(torch.cat([pre_morpho, upos_emb], dim=-1))
        xpos = self._xpos(torch.cat([pre_morpho, upos_emb], dim=-1))

        # parsing
        word_emb_ext = torch.cat(
            [torch.zeros((word_emb_ext.shape[0], 1, self._config.external_proj_size), device=self._get_device(),
                         dtype=torch.float), word_emb_ext], dim=1)
        x = mask_concat([x_parse, word_emb_ext], 0.33, self.training, self._get_device())
        x = torch.cat([x, lang_emb], dim=-1)
        output, _ = self._rnn(x)
        output = torch.cat([output, lang_emb], dim=-1)
        pre_parsing = torch.dropout(torch.tanh(self._pre_out(output)), 0.33, self.training)
        # h_r1 = torch.tanh(self._head_r1(pre_parsing))
        # h_r2 = torch.tanh(self._head_r2(pre_parsing))
        # l_r1 = torch.tanh(self._label_r1(pre_parsing))
        # l_r2 = torch.tanh(self._label_r2(pre_parsing))
        # att_stack = []
        # for ii in range(1, h_r1.shape[1]):
        #     a = self._att_net(h_r1[:, ii, :], h_r2)
        #     att_stack.append(a.unsqueeze(1))
        # att = torch.cat(att_stack, dim=1)
        heads = self._head(pre_parsing, pre_parsing)
        labels = pre_parsing  # self._label(pre_parsing, pre_parsing)
        return heads.squeeze(-1)[:, 1:, :], labels, upos, xpos, attrs, aupos, axpos, aattrs

    def _get_labels(self, labels, heads):
        x1 = labels
        labs = []
        for ii in range(labels.shape[0]):
            lab = []
            for jj in range(1, labels.shape[1]):
                if jj <= len(heads[ii]):
                    lab.append(labels[ii, heads[ii][jj - 1]].unsqueeze(0).unsqueeze(0))
                else:
                    lab.append(labels[ii, 0].unsqueeze(0).unsqueeze(0))
            lab = torch.cat(lab, dim=1)
            labs.append(lab)
        x2 = torch.cat(labs, dim=0)

        labs = self._label(x1[:, 1:, :], x2)
        return labs

    # def _get_labels(self, x1, x2, heads):
    #     x1 = x1[:, 1:, :]
    #     x_stack = []
    #     for ii in range(x1.shape[0]):
    #         xx = []
    #         for jj in range(x1.shape[1]):
    #             if jj < len(heads[ii]):
    #                 xx.append(x2[ii, heads[ii][jj]].unsqueeze(0).unsqueeze(0))
    #             else:
    #                 xx.append(x2[ii, 0].unsqueeze(0).unsqueeze(0))
    #         x_stack.append(torch.cat(xx, dim=1))
    #     x_stack = torch.cat(x_stack, dim=0).contiguous()
    #     x1 = x1.contiguous()
    #     hid = torch.cat([x1, x_stack], dim=-1)
    #     return self._label_linear(hid) + self._label_bilinear(x1, x_stack)

    def _get_device(self):
        if self._lang_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._lang_emb.weight.device.type, str(self._lang_emb.weight.device.index))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        att, labels, p_upos, p_xpos, p_attrs, a_upos, a_xpos, a_attrs = self.forward(batch)
        y_upos = batch['y_upos']
        y_attrs = batch['y_attrs']
        y_xpos = batch['y_xpos']
        y_head = batch['y_head']
        y_label = batch['y_label']

        pred_labels = self._get_labels(labels, y_head.detach().cpu().numpy())

        loss_upos = F.cross_entropy(p_upos.view(-1, p_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_xpos = F.cross_entropy(p_xpos.view(-1, p_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_attrs = F.cross_entropy(p_attrs.view(-1, p_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)

        loss_aupos = F.cross_entropy(a_upos.view(-1, a_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_axpos = F.cross_entropy(a_xpos.view(-1, a_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_aattrs = F.cross_entropy(a_attrs.view(-1, a_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)

        loss_uas = F.cross_entropy(att.reshape(-1, att.shape[2]), y_head.view(-1))
        loss_las = F.cross_entropy(pred_labels.view(-1, pred_labels.shape[2]), y_label.view(-1), ignore_index=0)

        step_loss = loss_uas + loss_las + (((loss_upos + loss_attrs + loss_xpos) / 3.) + (
                (loss_aupos + loss_aattrs + loss_axpos) / 3.))

        return {'loss': step_loss}

    def validation_step(self, batch, batch_idx):
        y_upos = batch['y_upos']
        del batch['y_upos']
        att, labels, p_upos, p_xpos, p_attrs, a_upos, a_xpos, a_attrs = self.forward(batch)
        y_xpos = batch['y_xpos']
        y_attrs = batch['y_attrs']
        y_head = batch['y_head']
        y_label = batch['y_label']
        x_sent_len = batch['x_sent_len']
        x_lang = batch['x_lang_sent']
        sl = x_sent_len.detach().cpu().numpy()

        att = torch.softmax(att, dim=-1).detach().cpu().numpy()
        pred_heads = self._decoder.decode(att, sl)
        pred_labels = self._get_labels(labels, pred_heads)

        loss_upos = F.cross_entropy(p_upos.view(-1, p_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_attrs = F.cross_entropy(p_attrs.view(-1, p_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)
        loss = (loss_upos + loss_attrs) / 3
        language_result = {lang_id: {'total': 0, 'upos_ok': 0, 'xpos_ok': 0, 'attrs_ok': 0, 'uas_ok': 0, 'las_ok': 0}
                           for lang_id in range(self._num_langs)}

        pred_upos = torch.argmax(p_upos, dim=-1).detach().cpu().numpy()
        pred_attrs = torch.argmax(p_attrs, dim=-1).detach().cpu().numpy()
        pred_xpos = torch.argmax(p_xpos, dim=-1).detach().cpu().numpy()
        pred_labels = torch.argmax(pred_labels, dim=-1).detach().cpu().numpy()
        tar_upos = y_upos.detach().cpu().numpy()
        tar_attrs = y_attrs.detach().cpu().numpy()
        tar_xpos = y_xpos.detach().cpu().numpy()
        tar_head = y_head.detach().cpu().numpy()
        tar_label = y_label.detach().cpu().numpy()

        x_lang = x_lang.detach().cpu().numpy()
        for iSent in range(p_upos.shape[0]):
            for iWord in range(sl[iSent]):
                lang_id = x_lang[iSent] - 1
                language_result[lang_id]['total'] += 1
                if pred_upos[iSent, iWord] == tar_upos[iSent, iWord]:
                    language_result[lang_id]['upos_ok'] += 1
                if pred_attrs[iSent, iWord] == tar_attrs[iSent, iWord]:
                    language_result[lang_id]['attrs_ok'] += 1
                if pred_xpos[iSent, iWord] == tar_xpos[iSent, iWord]:
                    language_result[lang_id]['xpos_ok'] += 1
                if pred_heads[iSent][iWord] == tar_head[iSent, iWord]:
                    language_result[lang_id]['uas_ok'] += 1
                    if pred_labels[iSent, iWord] == tar_label[iSent, iWord]:
                        language_result[lang_id]['las_ok'] += 1

        return {'loss': loss, 'acc': language_result}

    def validation_epoch_end(self, outputs):
        language_result = {lang_id: {'total': 0, 'upos_ok': 0, 'attrs_ok': 0, 'xpos_ok': 0, 'uas_ok': 0, 'las_ok': 0}
                           for lang_id in
                           range(self._num_langs)}

        valid_loss_total = 0
        total = 0
        attrs_ok = 0
        upos_ok = 0
        xpos_ok = 0
        uas_ok = 0
        las_ok = 0
        for out in outputs:
            valid_loss_total += out['loss']
            for lang_id in language_result:
                valid_loss_total += out['loss']
                language_result[lang_id]['total'] += out['acc'][lang_id]['total']
                language_result[lang_id]['upos_ok'] += out['acc'][lang_id]['upos_ok']
                language_result[lang_id]['xpos_ok'] += out['acc'][lang_id]['xpos_ok']
                language_result[lang_id]['attrs_ok'] += out['acc'][lang_id]['attrs_ok']
                language_result[lang_id]['uas_ok'] += out['acc'][lang_id]['uas_ok']
                language_result[lang_id]['las_ok'] += out['acc'][lang_id]['las_ok']
                # global
                total += out['acc'][lang_id]['total']
                upos_ok += out['acc'][lang_id]['upos_ok']
                xpos_ok += out['acc'][lang_id]['xpos_ok']
                attrs_ok += out['acc'][lang_id]['attrs_ok']
                uas_ok += out['acc'][lang_id]['uas_ok']
                las_ok += out['acc'][lang_id]['las_ok']

        self.log('val/loss', valid_loss_total / len(outputs))
        self.log('val/UPOS/total', upos_ok / total)
        self.log('val/ATTRS/total', attrs_ok / total)
        self.log('val/XPOS/total', xpos_ok / total)
        self.log('val/UAS/total', uas_ok / total)
        self.log('val/LAS/total', las_ok / total)

        res = {}
        for lang_index in language_result:
            total = language_result[lang_index]['total']
            if total == 0:
                total = 1
            if self._language_codes is None:
                lang = lang_index
            else:
                lang = self._language_codes[lang_index]
            res[lang] = {
                "upos": language_result[lang_index]['upos_ok'] / total,
                "xpos": language_result[lang_index]['xpos_ok'] / total,
                "attrs": language_result[lang_index]['attrs_ok'] / total,
                "uas": language_result[lang_index]['uas_ok'] / total,
                "las": language_result[lang_index]['las_ok'] / total
            }

            self.log('val/UPOS/{0}'.format(lang), language_result[lang_index]['upos_ok'] / total)
            self.log('val/XPOS/{0}'.format(lang), language_result[lang_index]['xpos_ok'] / total)
            self.log('val/ATTRS/{0}'.format(lang), language_result[lang_index]['attrs_ok'] / total)
            self.log('val/UAS/{0}'.format(lang), language_result[lang_index]['uas_ok'] / total)
            self.log('val/LAS/{0}'.format(lang), language_result[lang_index]['las_ok'] / total)

        # single value for early stopping
        self._epoch_results = self._compute_early_stop(res)
        self.log('val/early_meta', self._early_stop_meta_val)

    def load(self, model_path: str, device: str = 'cpu'):
        self.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
        self.to(device)

    def process(self, doc: Document, collate: MorphoCollate, batch_size: int = 4, num_workers: int = 4) -> Document:
        self.eval()
        dataset = MorphoDataset(doc)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate.collate_fn,
                                shuffle=False, num_workers=num_workers, pin_memory=True)
        index = 0
        with torch.no_grad():
            for batch in dataloader:

                del batch['y_upos']
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self._device)
                att, labels, p_upos, p_xpos, p_attrs, a_upos, a_xpos, a_attrs = self.forward(batch)

                x_sent_len = batch['x_sent_len']
                sl = x_sent_len.detach().cpu().numpy()

                batch_size = p_upos.size()[0]

                att = torch.softmax(att, dim=-1).detach().cpu().numpy()
                pred_heads = self._decoder.decode(att, sl)
                pred_labels = self._get_labels(labels, pred_heads)
                pred_labels = torch.argmax(pred_labels.detach(), dim=-1).cpu()
                p_upos = torch.argmax(p_upos, dim=-1).detach().cpu().numpy()
                p_xpos = torch.argmax(p_xpos, dim=-1).detach().cpu().numpy()
                p_attrs = torch.argmax(p_attrs, dim=-1).detach().cpu().numpy()
                for sentence_index in range(batch_size):  # for each sentence
                    # print(f"at index {index+sentence_index}, sentence {sentence_index} has {batch['x_sent_len'][sentence_index]} words.")
                    for word_index in range(batch["x_sent_len"][sentence_index]):
                        head = pred_heads[sentence_index][word_index]
                        label_id = pred_labels[sentence_index][word_index]
                        doc.sentences[index + sentence_index].words[word_index].head = head
                        doc.sentences[index + sentence_index].words[word_index].label = self._encodings.labels[label_id]
                        doc.sentences[index + sentence_index].words[word_index].upos = self._encodings.upos_list[
                            p_upos[sentence_index, word_index]]
                        doc.sentences[index + sentence_index].words[word_index].xpos = self._encodings.xpos_list[
                            p_xpos[sentence_index, word_index]]
                        doc.sentences[index + sentence_index].words[word_index].attrs = self._encodings.attrs_list[
                            p_attrs[sentence_index, word_index]]

                index += batch_size

        return doc

    class PrintAndSaveCallback(pl.callbacks.Callback):
        def __init__(self, store_prefix):
            super().__init__()
            self.store_prefix = store_prefix

        def on_validation_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            epoch = trainer.current_epoch

            for lang in pl_module._epoch_results:
                res = pl_module._epoch_results[lang]
                if "uas_best" in res:
                    trainer.save_checkpoint(self.store_prefix + "." + lang + ".uas")
                if "las_best" in res:
                    trainer.save_checkpoint(self.store_prefix + "." + lang + ".las")

            trainer.save_checkpoint(self.store_prefix + ".last")

            s = "{0:30s}\tUAS\tLAS\tUPOS\tXPOS\tATTRS".format("Language")
            print("\n\n\t" + s)
            print("\t" + ("=" * (len(s) + 16)))
            for lang in pl_module._language_codes:
                uas = metrics["val/UAS/{0}".format(lang)]
                las = metrics["val/LAS/{0}".format(lang)]
                upos = metrics["val/UPOS/{0}".format(lang)]
                xpos = metrics["val/XPOS/{0}".format(lang)]
                attrs = metrics["val/ATTRS/{0}".format(lang)]
                msg = "\t{0:30s}:\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}".format(lang, uas, las, upos, xpos,
                                                                                       attrs)
                print(msg)
            print("\n")
