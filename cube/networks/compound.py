import sys
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader

from cube.io_utils.objects import Document
from cube.networks.utils import Word2TargetCollate, CompoundDataset

sys.path.append('')
from cube.io_utils.encodings import Encodings
from cube.io_utils.config import CompoundConfig
from cube.networks.modules import LinearNorm, ConvNorm, Attention


class Compound(pl.LightningModule):
    encodings: Encodings
    config: CompoundConfig

    def __init__(self, config: CompoundConfig, encodings: Encodings, language_codes: [] = None):
        super(Compound, self).__init__()
        NUM_FILTERS = 512
        NUM_LAYERS = 5
        self._config = config
        self._encodings = encodings
        self._num_languages = encodings.num_langs
        self._language_codes = language_codes
        self._eol = len(encodings.char2int)
        self._num_filters = NUM_FILTERS

        self._char_list = ['' for char in encodings.char2int]
        for char in encodings.char2int:
            self._char_list[encodings.char2int[char]] = char
        self._lang_emb = nn.Embedding(self._num_languages + 1, config.lang_emb_size, padding_idx=0)
        self._char_emb = nn.Embedding(len(encodings.char2int) + 2, config.char_emb_size,
                                      padding_idx=0)  # start/stop index
        self._case_emb = nn.Embedding(4, 16, padding_idx=0)  # 0-pad 1-symbol 2-upper 3-lower
        convolutions = []
        cs_inp = config.char_emb_size + config.lang_emb_size + 16

        for _ in range(NUM_LAYERS):
            conv_layer = nn.Sequential(
                ConvNorm(cs_inp,
                         NUM_FILTERS,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(NUM_FILTERS))
            convolutions.append(conv_layer)
            cs_inp = NUM_FILTERS // 2 + config.lang_emb_size

        self._convolutions_char = nn.ModuleList(convolutions)
        self._decoder = nn.LSTM(
            NUM_FILTERS // 2 + config.char_emb_size + config.lang_emb_size + 16,
            config.decoder_size, config.decoder_layers,
            batch_first=True, bidirectional=False)
        self._attention = Attention(
            (NUM_FILTERS // 2 + config.lang_emb_size + 16) // 2,
            config.decoder_size, config.att_proj_size)

        self._output_char = LinearNorm(config.decoder_size, len(self._encodings.char2int) + 2)
        self._output_case = LinearNorm(config.decoder_size, 4)
        self._start_frame = nn.Embedding(1,
                                         NUM_FILTERS // 2 + config.char_emb_size + config.lang_emb_size + 16)

        if self._language_codes:
            self._res = {}
            for language_code in self._language_codes:
                self._res[language_code] = {"loss": 0., "acc": 0.}
            self._early_stop_meta_val = 0
            self._epoch_results = None

    def forward(self, X):
        x_char = X['x_char']
        x_case = X['x_case']
        x_lang = X['x_lang']
        x_upos = X['x_upos']

        if 'y_char' in X:
            gs_output = X['y_char']
        else:
            gs_output = None

        char_emb = self._char_emb(x_char)
        case_emb = self._case_emb(x_case)

        lang_emb = self._lang_emb(x_lang).unsqueeze(1).repeat(1, char_emb.shape[1], 1)
        conditioning = case_emb
        if gs_output is not None:
            output_idx = gs_output

        x = torch.cat((char_emb, conditioning), dim=-1)
        half = self._num_filters // 2
        count = 0
        res = None
        skip = None
        x_lang_conv = lang_emb.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        for conv in self._convolutions_char:
            count += 1
            drop = self.training
            if count >= len(self._convolutions_char):
                drop = False
            if skip is not None:
                x = x + skip

            x = torch.cat([x, x_lang_conv], dim=1)
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
        encoder_output = torch.cat((x, conditioning, lang_emb), dim=-1)

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
            out_char_list.append(output_char)
            out_case_list.append(output_case)
            selected_chars = torch.argmax(output_char, dim=-1)
            for ii in range(selected_chars.shape[0]):
                if selected_chars[ii].squeeze() == self._eol:
                    done[ii] = 1
            if gs_output is not None:
                prev_char_emb = self._char_emb(output_idx[:, step]).unsqueeze(1)
            else:
                prev_char_emb = self._char_emb(selected_chars)

            step += 1

        return torch.cat(out_char_list, dim=1), torch.cat(out_case_list, dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, model_path: str, device: str = 'cpu'):
        self.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
        self.to(device)

    def _get_device(self):
        if self._char_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._char_emb.weight.device.type, str(self._char_emb.weight.device.index))

    def process(self, doc: Document, collate: Word2TargetCollate, batch_size: int = 4,
                num_workers: int = 4) -> Document:
        self.eval()
        dataset = CompoundDataset(doc, for_training=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate.collate_fn,
                                shuffle=False, num_workers=num_workers, pin_memory=True)

        data_iterator = iter(dataloader)

        end_char_value = len(self._encodings.char2int)

        with torch.no_grad():
            all_lemmas = []
            for batch in dataloader:
                del batch['y_char']  # set for prediction, not training
                del batch['y_case']

                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self._device)

                y_char_pred, y_case_pred = self.forward(batch)
                y_char_pred = torch.argmax(y_char_pred.detach(), dim=-1).cpu().numpy()  # list of lists of int
                y_case_pred = torch.argmax(y_case_pred.detach(), dim=-1).cpu().numpy()  # list of lists of int
                for word_index in range(y_char_pred.shape[0]):
                    # get letters
                    lemma = []
                    for char_val, case_val in zip(y_char_pred[word_index],
                                                  y_case_pred[word_index]):  # [[24, 12, 88]], get the inside list
                        if char_val == end_char_value:
                            break
                        chr = self._encodings.characters[char_val]
                        if case_val == 2:
                            chr = chr.upper()
                        elif case_val == 3:
                            chr = chr.lower()
                        lemma.append(chr)

                    all_lemmas.append("".join(lemma))
            compound_index = 0
            for sentence_index in range(len(doc.sentences)):
                for word_index in range(len(doc.sentences[sentence_index].words)):
                    if doc.sentences[sentence_index].words[word_index].is_compound_entry:
                        from ipdb import set_trace
                        set_trace()
                    # doc.sentences[sentence_index].words[word_index].lemma = all_lemmas[lemma_index]
                    compound_index += 1
        return doc

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

    def training_step(self, batch, batch_idx):
        y_char_pred, y_case_pred = self.forward(batch)
        y_char_target, y_case_target = batch['y_char'], batch['y_case']
        loss_char = F.cross_entropy(y_char_pred.view(-1, y_char_pred.shape[2]), y_char_target.view(-1), ignore_index=0)
        loss_case = F.cross_entropy(y_case_pred.view(-1, y_case_pred.shape[2]), y_case_target.view(-1), ignore_index=0)
        return loss_char + loss_case

    def validation_step(self, batch, batch_idx):
        y_char_target, y_case_target = batch['y_char'], batch['y_case']
        del batch['y_char']
        y_char_pred, y_case_pred = self.forward(batch)
        language_result = {lang_id: {'total': 0, 'ok': 0}
                           for lang_id in range(self._num_languages)}

        y_char_target = y_char_target.detach().cpu().numpy()
        y_char_pred = torch.argmax(y_char_pred.detach(), dim=-1).cpu().numpy()
        lang = batch['x_lang'].detach().cpu().numpy()
        for lang_id, y_pred, y_target in zip(lang, y_char_pred, y_char_target):
            valid = True
            for y_p, y_t in zip(y_pred, y_target):
                if y_t != 0 and y_p != y_t:
                    valid = False
                    break
            if valid:
                language_result[lang_id - 1]['ok'] += 1
            language_result[lang_id - 1]['total'] += 1

        return {'acc': language_result}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        language_result = {lang_id: {'total': 0, 'ok': 0}
                           for lang_id in
                           range(self._num_languages)}
        for result in outputs:
            for lang_id in result['acc']:
                language_result[lang_id]['ok'] += result['acc'][lang_id]['ok']
                language_result[lang_id]['total'] += result['acc'][lang_id]['total']

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
                "acc": language_result[lang_index]['ok'] / total,
            }

            self.log('val/ACC/{0}'.format(lang), language_result[lang_index]['ok'] / total)

        # single value for early stopping
        self._epoch_results = self._compute_early_stop(res)
        self.log('val/early_meta', self._early_stop_meta_val)

    def _compute_early_stop(self, res):
        for lang in res:
            if res[lang]["acc"] > self._res[lang]["acc"]:
                self._early_stop_meta_val += 1
                self._res[lang]["acc"] = res[lang]["acc"]
                res[lang]["acc_best"] = True
        return res

    class PrintAndSaveCallback(pl.callbacks.Callback):
        def __init__(self, store_prefix):
            super().__init__()
            self.store_prefix = store_prefix

        def on_validation_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            epoch = trainer.current_epoch

            for lang in pl_module._epoch_results:
                res = pl_module._epoch_results[lang]
                if "acc_best" in res:
                    trainer.save_checkpoint(self.store_prefix + "." + lang + ".best")

            trainer.save_checkpoint(self.store_prefix + ".last")

            s = "{0:30s}\tACC".format("Language")
            print("\n\n\t" + s)
            print("\t" + ("=" * (len(s) + 16)))
            for lang in pl_module._language_codes:
                acc = metrics["val/ACC/{0}".format(lang)]
                msg = "\t{0:30s}:\t{1:.4f}".format(lang, acc)
                print(msg)
            print("\n")
