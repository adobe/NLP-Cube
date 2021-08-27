import sys
sys.path.append('')
import os, yaml
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from cube.io_utils.objects import Document
from cube.io_utils.encodings import Encodings
from cube.io_utils.config import TaggerConfig
from cube.networks.modules import ConvNorm, LinearNorm, MLP
from cube.networks.utils import MorphoCollate, MorphoDataset, unpack, mask_concat
from cube.networks.modules import WordGram

class Tagger(pl.LightningModule):
    def __init__(self, config: TaggerConfig, encodings: Encodings, language_codes: [] = None, ext_word_emb=0):
        super().__init__()
        self._device = "cpu"  # default
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

        ext2int = []
        for input_size in self._ext_word_emb:
            module = MLP(input_size, config.external_proj_size)
            ext2int.append(module)
        self._ext_proj = nn.ModuleList(ext2int)
        self._word_emb = nn.Embedding(len(encodings.word2int), config.word_emb_size, padding_idx=0)
        self._lang_emb = nn.Embedding(encodings.num_langs + 1, config.lang_emb_size, padding_idx=0)
        self._convs = nn.ModuleList(conv_layers)
        self._upos = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.upos2int))
        self._upos_emb = nn.Embedding(len(encodings.upos2int), 64)
        self._xpos = LinearNorm(64 + config.lang_emb_size + NUM_FILTERS // 2, len(encodings.xpos2int))
        self._attrs = LinearNorm(64 + config.lang_emb_size + NUM_FILTERS // 2, len(encodings.attrs2int))

        self._aupos = LinearNorm(config.char_filter_size // 2 + config.lang_emb_size, len(encodings.upos2int))
        self._axpos = LinearNorm(config.char_filter_size // 2 + config.lang_emb_size, len(encodings.xpos2int))
        self._aattrs = LinearNorm(config.char_filter_size // 2 + config.lang_emb_size, len(encodings.attrs2int))

        if self._language_codes:
            self._res = {}
            for language_code in self._language_codes:
                self._res[language_code] = {"upos": 0., "xpos": 0., "attrs": 0.}
            self._early_stop_meta_val = 0

    def _compute_early_stop(self, res):
        for lang in res:
            if res[lang]["upos"] > self._res[lang]["upos"]:
                self._early_stop_meta_val += 1
                self._res[lang]["upos"] = res[lang]["upos"]
                res[lang]["upos_best"] = True
            if res[lang]["xpos"] > self._res[lang]["xpos"]:
                self._early_stop_meta_val += 1
                self._res[lang]["xpos"] = res[lang]["xpos"]
                res[lang]["xpos_best"] = True
            if res[lang]["attrs"] > self._res[lang]["attrs"]:
                self._early_stop_meta_val += 1
                self._res[lang]["attrs"] = res[lang]["attrs"]
                res[lang]["attrs_best"] = True
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
        char_emb_packed = self._word_net(x_words_chars, x_words_case, x_lang_word, x_word_masks, x_word_len)
        gs_upos = None
        if 'y_upos' in X:
            gs_upos = X['y_upos']
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
        lang_emb = lang_emb.unsqueeze(1).repeat(1, char_emb.shape[1], 1)

        aupos = self._aupos(torch.cat([char_emb, lang_emb], dim=-1))
        axpos = self._axpos(torch.cat([char_emb, lang_emb], dim=-1))
        aattrs = self._aattrs(torch.cat([char_emb, lang_emb], dim=-1))

        word_emb = self._word_emb(x_sents)

        x = mask_concat([word_emb, char_emb, word_emb_ext], 0.33, self.training, self._get_device())
        x = torch.cat([x, lang_emb], dim=-1)
        x = x.permute(0, 2, 1)
        lang_emb = lang_emb.permute(0, 2, 1)
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
                x = torch.cat([x, lang_emb], dim=1)
        x = x + res
        xu = x.permute(0, 2, 1).contiguous()
        x = torch.cat([x, lang_emb], dim=1)
        x = x.permute(0, 2, 1)
        # x = torch.tanh(x)
        upos = self._upos(x)
        if gs_upos is None:
            upos_idx = torch.argmax(upos, dim=-1)
        else:
            upos_idx = gs_upos
        upos_emb = self._upos_emb(upos_idx)
        upos_emb = torch.cat([upos_emb, lang_emb.permute(0, 2, 1)], dim=-1)
        xpos = self._xpos(torch.cat([upos_emb, xu], dim=-1))
        attrs = self._attrs(torch.cat([upos_emb, xu], dim=-1))
        return upos, xpos, attrs, aupos, axpos, aattrs

    def _get_device(self):
        if self._lang_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._lang_emb.weight.device.type, str(self._lang_emb.weight.device.index))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        p_upos, p_xpos, p_attrs, a_upos, a_xpos, a_attrs = self.forward(batch)
        y_upos = batch['y_upos']
        y_xpos = batch['y_xpos']
        y_attrs = batch['y_attrs']

        loss_upos = F.cross_entropy(p_upos.view(-1, p_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_xpos = F.cross_entropy(p_xpos.reshape(-1, p_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_attrs = F.cross_entropy(p_attrs.reshape(-1, p_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)

        loss_aupos = F.cross_entropy(a_upos.view(-1, a_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_axpos = F.cross_entropy(a_xpos.view(-1, a_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_aattrs = F.cross_entropy(a_attrs.view(-1, a_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)

        step_loss = ((loss_upos + loss_attrs + loss_xpos) / 3.) * 1.0 + (
                (loss_aupos + loss_aattrs + loss_axpos) / 3.) * 1.0

        return {'loss': step_loss}

    def validation_step(self, batch, batch_idx):
        y_upos = batch['y_upos']
        y_xpos = batch['y_xpos']
        y_attrs = batch['y_attrs']
        x_sent_len = batch['x_sent_len']
        x_lang = batch['x_lang_sent']
        del batch['y_upos']
        p_upos, p_xpos, p_attrs, a_upos, a_xpos, a_attrs = self.forward(batch)

        loss_upos = F.cross_entropy(p_upos.view(-1, p_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_xpos = F.cross_entropy(p_xpos.reshape(-1, p_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_attrs = F.cross_entropy(p_attrs.reshape(-1, p_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)
        loss = (loss_upos + loss_attrs + loss_xpos) / 3
        language_result = {lang_id: {'total': 0, 'upos_ok': 0, 'xpos_ok': 0, 'attrs_ok': 0} for lang_id in
                           range(self._num_langs)}

        pred_upos = torch.argmax(p_upos, dim=-1).detach().cpu().numpy()
        pred_xpos = torch.argmax(p_xpos, dim=-1).detach().cpu().numpy()
        pred_attrs = torch.argmax(p_attrs, dim=-1).detach().cpu().numpy()
        tar_upos = y_upos.detach().cpu().numpy()
        tar_xpos = y_xpos.detach().cpu().numpy()
        tar_attrs = y_attrs.detach().cpu().numpy()
        sl = x_sent_len.detach().cpu().numpy()
        x_lang = x_lang.detach().cpu().numpy()
        for iSent in range(p_upos.shape[0]):
            for iWord in range(sl[iSent]):
                lang_id = x_lang[iSent] - 1
                language_result[lang_id]['total'] += 1
                if pred_upos[iSent, iWord] == tar_upos[iSent, iWord]:
                    language_result[lang_id]['upos_ok'] += 1
                if pred_xpos[iSent, iWord] == tar_xpos[iSent, iWord]:
                    language_result[lang_id]['xpos_ok'] += 1
                if pred_attrs[iSent, iWord] == tar_attrs[iSent, iWord]:
                    language_result[lang_id]['attrs_ok'] += 1

        return {'loss': loss, 'acc': language_result}

    def validation_epoch_end(self, outputs):
        language_result = {lang_index: {'total': 0, 'upos_ok': 0, 'xpos_ok': 0, 'attrs_ok': 0} for lang_index in
                           range(self._num_langs)}

        valid_loss_total = 0
        total = 0
        attrs_ok = 0
        upos_ok = 0
        xpos_ok = 0
        for out in outputs:
            valid_loss_total += out['loss']
            for lang_index in language_result:
                valid_loss_total += out['loss']
                language_result[lang_index]['total'] += out['acc'][lang_index]['total']
                language_result[lang_index]['upos_ok'] += out['acc'][lang_index]['upos_ok']
                language_result[lang_index]['xpos_ok'] += out['acc'][lang_index]['xpos_ok']
                language_result[lang_index]['attrs_ok'] += out['acc'][lang_index]['attrs_ok']
                # global
                total += out['acc'][lang_index]['total']
                upos_ok += out['acc'][lang_index]['upos_ok']
                xpos_ok += out['acc'][lang_index]['xpos_ok']
                attrs_ok += out['acc'][lang_index]['attrs_ok']

        self.log('val/loss', valid_loss_total / len(outputs))
        self.log('val/UPOS/total', upos_ok / total)
        self.log('val/XPOS/total', xpos_ok / total)
        self.log('val/ATTRS/total', attrs_ok / total)

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
                "attrs": language_result[lang_index]['attrs_ok'] / total
            }

            self.log('val/UPOS/{0}'.format(lang), language_result[lang_index]['upos_ok'] / total)
            self.log('val/XPOS/{0}'.format(lang), language_result[lang_index]['xpos_ok'] / total)
            self.log('val/ATTRS/{0}'.format(lang), language_result[lang_index]['attrs_ok'] / total)

        # single value for early stopping
        self._epoch_results = self._compute_early_stop(res)
        self.log('val/early_meta', self._early_stop_meta_val)

        # print("\n\n\n", upos_ok / total, xpos_ok / total, attrs_ok / total,
        #      aupos_ok / total, axpos_ok / total, aattrs_ok / total, "\n\n\n")

    def load(self, model_path:str, device: str = 'cpu'):
        self.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
        self.to(device)

    def process(self, doc: Document, collate: MorphoCollate, upos: bool = True, xpos: bool = True, attrs: bool = True,
                batch_size: int = 32, num_workers: int = 4) -> Document:
        self.eval()
        if not (upos or xpos or attrs):
            raise Exception("To perform tagging at least one of 'upos', 'xpos' or 'attrs' must be set to True.")

        dataset = MorphoDataset(doc)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate.collate_fn,
                                shuffle=False, num_workers=num_workers, pin_memory=True)
        index = 0

        with torch.no_grad():
            for batch in dataloader:
                del batch['y_upos']
                p_upos, p_xpos, p_attrs, _, _, _ = self.forward(batch)

                batch_size = p_upos.size()[0]

                if upos:
                    pred_upos = torch.argmax(p_upos, dim=-1).detach().cpu().numpy()
                if xpos:
                    pred_xpos = torch.argmax(p_xpos, dim=-1).detach().cpu().numpy()
                if attrs:
                    pred_attrs = torch.argmax(p_attrs, dim=-1).detach().cpu().numpy()

                for sentence_index in range(batch_size):  # for each sentence
                    # print(f"at index {index+sentence_index}, sentence {sentence_index} has {batch['x_sent_len'][sentence_index]} words.")
                    for word_index in range(batch["x_sent_len"][sentence_index]):
                        if upos:
                            doc.sentences[index + sentence_index].words[word_index].upos = self._encodings.upos_list[
                                pred_upos[sentence_index][word_index]]
                        if xpos:
                            doc.sentences[index + sentence_index].words[word_index].xpos = self._encodings.xpos_list[
                                pred_xpos[sentence_index][word_index]]
                        if attrs:
                            doc.sentences[index + sentence_index].words[word_index].attrs = self._encodings.attrs_list[
                                pred_attrs[sentence_index][word_index]]

                index += batch_size
        #                break
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
                if "upos_best" in res:
                    trainer.save_checkpoint(self.store_prefix + "." + lang + ".upos")
                if "xpos_best" in res:
                    trainer.save_checkpoint(self.store_prefix + "." + lang + ".xpos")
                if "attrs_best" in res:
                    trainer.save_checkpoint(self.store_prefix + "." + lang + ".attrs")

            trainer.save_checkpoint(self.store_prefix + ".last")

            s = "{0:30s}\tUPOS\tXPOS\tATTRS".format("Language")
            print("\n\n\t" + s)
            print("\t" + ("=" * (len(s) + 9)))
            for lang in pl_module._language_codes:
                upos = metrics["val/UPOS/{0}".format(lang)]
                xpos = metrics["val/XPOS/{0}".format(lang)]
                attrs = metrics["val/ATTRS/{0}".format(lang)]
                msg = "\t{0:30s}:\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(lang, upos, xpos, attrs)
                print(msg)
            print("\n")


if __name__ == '__main__':

    root = "data/be"
    language_code = "be_hse"
    device = 'cpu'  # 'cuda'
    batch_size = 2

    # read yaml
    object_config = yaml.full_load(open(root + ".yaml"))

    # read model config
    config = TaggerConfig(filename=root + ".config")

    # read encodings
    encodings = Encodings()
    encodings.load(filename=root + ".encodings")

    # load models
    tagger_UPOS = Tagger.load_from_checkpoint(root + "." + language_code + ".upos", config=config, encodings=encodings,
                                              language_codes=object_config["language_codes"])
    tagger_UPOS.to(device)
    tagger_UPOS.eval()
    tagger_UPOS.freeze()

    tagger_XPOS = Tagger.load_from_checkpoint(root + "." + language_code + ".xpos", config=config, encodings=encodings,
                                              language_codes=object_config["language_codes"])
    tagger_XPOS.to(device)
    tagger_XPOS.eval()
    tagger_XPOS.freeze()

    tagger_ATTRS = Tagger.load_from_checkpoint(root + "." + language_code + ".attrs", config=config,
                                               encodings=encodings,
                                               language_codes=object_config["language_codes"])
    tagger_ATTRS.to(device)
    tagger_ATTRS.eval()
    tagger_ATTRS.freeze()

    # read a doc
    doc = Document()
    doc.load("corpus/ud-treebanks-v2.5/UD_Belarusian-HSE/be_hse-ud-dev.conllu",
             lang_id=object_config["language_codes"].index(language_code))

    for si, _ in enumerate(doc.sentences):
        for wi, _ in enumerate(doc.sentences[si].words):
            doc.sentences[si].words[wi].upos = "<change>"
            doc.sentences[si].words[wi].xpos = "<change>"
            doc.sentences[si].words[wi].attrs = "<change>"

    print(doc)

    doc = tagger_UPOS.process(doc, upos=True, xpos=False, attrs=False, batch_size=batch_size)
    doc = tagger_XPOS.process(doc, upos=False, xpos=True, attrs=False, batch_size=batch_size)
    doc = tagger_ATTRS.process(doc, upos=False, xpos=False, attrs=True, batch_size=batch_size)

    print(doc)
