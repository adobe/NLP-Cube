#
# Author: Tiberiu Boros
#
# Copyright (c) 2019 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import optparse
import sys
import random

sys.path.append('')
import numpy as np
import torch.nn as nn
import torch.utils.data
from cube.networks.text import TextEncoder
from cube.io_utils.config import ParserConfig
from cube.io_utils.encodings import Encodings
from cube.networks.modules import Attention
from cube.networks.decoders import GreedyDecoder


def nll_loss(logs, targets):
    out = torch.diag(logs[:, targets])
    return -torch.mean(out)


class Parser(nn.Module):
    encodings: Encodings
    config: ParserConfig

    def __init__(self, config, encodings, num_languages=1, target_device='cpu'):
        super(Parser, self).__init__()
        self.config = config
        self.encodings = encodings
        self.num_languages = num_languages
        self._target_device = target_device
        self._decoder = GreedyDecoder()
        lang_emb_size = self.config.tagger_embeddings_size
        self.lang_emb = nn.Embedding(num_languages, lang_emb_size)

        self.text_network = TextEncoder(config, encodings, ext_conditioning=lang_emb_size, target_device=target_device,
                                        nn_type=nn.LSTM)

        self.proj_arc_head = nn.Sequential(
            nn.Linear(self.config.tagger_mlp_layer + lang_emb_size, self.config.parser_arc_proj_size), nn.Tanh(),
            nn.Dropout(self.config.tagger_mlp_dropout))
        self.proj_arc_dep = nn.Sequential(
            nn.Linear(self.config.tagger_mlp_layer + lang_emb_size, self.config.parser_arc_proj_size), nn.Tanh(),
            nn.Dropout(self.config.tagger_mlp_dropout))
        self.proj_label_head = nn.Sequential(
            nn.Linear(self.config.tagger_mlp_layer + lang_emb_size, self.config.parser_label_proj_size), nn.Tanh(),
            nn.Dropout(self.config.tagger_mlp_dropout))
        self.proj_label_dep = nn.Sequential(
            nn.Linear(self.config.tagger_mlp_layer + lang_emb_size, self.config.parser_label_proj_size), nn.Tanh(),
            nn.Dropout(self.config.tagger_mlp_dropout))
        self.output_label = nn.Linear(self.config.parser_label_proj_size * 2 + lang_emb_size,
                                      len(self.encodings.label2int))
        # self.head_bias = nn.Linear(self.config.parser_arc_proj_size + lang_emb_size, 1)
        # self.dep_bias = nn.Linear(self.config.parser_arc_proj_size + lang_emb_size, 1)

        self.aux_mlp = nn.Sequential(
            nn.Linear(self.config.tagger_mlp_layer + lang_emb_size, self.config.tagger_mlp_layer),
            nn.Tanh(), nn.Dropout(p=self.config.tagger_mlp_dropout))
        self.aux_output_upos = nn.Linear(self.config.tagger_mlp_layer + lang_emb_size, len(self.encodings.upos2int))
        self.aux_output_xpos = nn.Linear(self.config.tagger_mlp_layer + lang_emb_size, len(self.encodings.xpos2int))
        self.aux_output_attrs = nn.Linear(self.config.tagger_mlp_layer + lang_emb_size, len(self.encodings.attrs2int))
        self.attention = Attention(self.config.parser_arc_proj_size // 2,
                                   self.config.parser_arc_proj_size + lang_emb_size)
        self.dropout = nn.Dropout(self.config.tagger_encoder_dropout)

    def forward(self, x, lang_ids=None, warmup=False):
        if lang_ids is not None and self.lang_emb is not None:
            lang_ids = torch.tensor(lang_ids, dtype=torch.long, device=self._get_device())
            lang_emb = self.lang_emb(lang_ids)
        else:
            lang_emb = None
        emb, hidden = self.text_network(x, conditioning=lang_emb)
        lang_emb_parsing = lang_emb.unsqueeze(1).repeat(1, emb.shape[1] + 1, 1)
        lang_emb = lang_emb.unsqueeze(1).repeat(1, emb.shape[1], 1)
        emb_output = torch.cat((emb, lang_emb), dim=2)

        proj_arc_head = self.proj_arc_head(emb_output)
        proj_label_head = self.proj_label_head(emb_output)
        proj_arc_dep = self.proj_arc_dep(emb_output)
        proj_label_dep = self.proj_label_dep(emb_output)
        w_stack = []
        proj_arc_head = torch.cat(
            (torch.zeros((proj_arc_head.shape[0], 1, proj_arc_head.shape[2]), device=self._get_device()),
             proj_arc_head), dim=1)
        proj_label_head = torch.cat(
            (torch.zeros((proj_label_head.shape[0], 1, proj_label_head.shape[2]), device=self._get_device()),
             proj_label_head), dim=1)

        proj_arc_dep = torch.cat(
            (torch.zeros((proj_arc_dep.shape[0], 1, proj_arc_dep.shape[2]), device=self._get_device()),
             proj_arc_dep), dim=1)
        proj_label_dep = torch.cat(
            (torch.zeros((proj_label_dep.shape[0], 1, proj_label_dep.shape[2]), device=self._get_device()),
             proj_label_dep), dim=1)

        proj_arc_head_lang = torch.cat((proj_arc_head, lang_emb_parsing), dim=2)  # .permute(1, 0, 2)
        proj_arc_dep = proj_arc_dep  # .permute((1, 0, 2))

        if not warmup:
            for ii in range(proj_arc_head_lang.shape[1]):
                att = self.attention(proj_arc_dep[:, ii, :], proj_arc_head_lang)
                w_stack.append(att.unsqueeze(1))

            arcs = torch.cat(w_stack, dim=1)  # .permute(1, 0, 2)
        # from ipdb import set_trace
        # set_trace()

        # head_bias = self.head_bias(torch.cat((proj_arc_dep, lang_emb_parsing.permute(1, 0, 2)), dim=2)).permute(1, 0, 2)
        # dep_bias = self.dep_bias(proj_arc_head_lang).permute(1, 0, 2)
        # dep_bias = dep_bias.unsqueeze(1).squeeze(3).repeat(1, dep_bias.shape[1], 1)
        # head_bias = head_bias.repeat(1, 1, head_bias.shape[1])
        # arcs = arcs + dep_bias
        aux_hid = self.aux_mlp(emb_output)
        s_aux_upos = self.aux_output_upos(torch.cat((aux_hid, lang_emb), dim=2))
        s_aux_xpos = self.aux_output_xpos(torch.cat((aux_hid, lang_emb), dim=2))
        s_aux_attrs = self.aux_output_attrs(torch.cat((aux_hid, lang_emb), dim=2))
        if warmup:
            out_arcs = None
        else:
            out_arcs = torch.log(arcs[:, 1:, :])
        return out_arcs, torch.cat((proj_label_head, lang_emb_parsing),
                                   dim=2), proj_label_dep, s_aux_upos, s_aux_xpos, s_aux_attrs

    def get_tree(self, arcs, lens, proj_label_head, proj_label_dep, gs_heads=None):
        if gs_heads is not None:
            heads = gs_heads
            max_sent_size = gs_heads.shape[1]
        else:
            heads = self._decoder.decode(np.exp(arcs), lens)
            max_sent_size = max([len(vect) for vect in heads])
        labels = []
        for idx_batch in range(proj_label_head.shape[0]):
            sent_labels = []
            if gs_heads is not None:
                s_len = proj_label_head.shape[1] - 1
            else:
                s_len = lens[idx_batch]
            for idx_word in range(s_len):
                if gs_heads is not None:
                    head_index = heads[idx_batch, idx_word]
                else:
                    head_index = heads[idx_batch][idx_word]
                hidden = torch.cat(
                    (proj_label_head[idx_batch, idx_word + 1, :], proj_label_dep[idx_batch, head_index, :]),
                    dim=0)
                sent_labels.append(hidden.unsqueeze(0))
            for _ in range(max_sent_size - s_len):
                sent_labels.append(hidden.unsqueeze(0))
            labels.append(torch.cat(sent_labels, dim=0).unsqueeze(0))
        return heads, self.output_label(torch.cat(labels, dim=0))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self._target_device))

    def _get_device(self):
        if self.text_network.i2h.linear_layer.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.text_network.i2h.linear_layer.weight.device.type,
                                str(self.text_network.i2h.linear_layer.weight.device.index))

    def process(self, sequences, lang_id):
        batch_size = 16
        self.eval()
        num_batches = len(sequences) // batch_size
        if len(sequences) % batch_size != 0:
            num_batches += 1
        total_words = 0

        import copy
        sent_id = 0
        word_id = 0
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            stop = min(len(sequences), start + batch_size)
            data = []
            lang_ids = []
            for ii in range(stop - start):
                data.append(sequences[start + ii])
                total_words += len(sequences[start + ii])
                lang_ids.append(lang_id)
            with torch.no_grad():
                new_seqs = []
                for seq in data:
                    new_seq = []
                    for elem in seq:
                        if not elem.is_compound_entry:
                            new_seq.append(elem)
                    new_seqs.append(new_seq)
                s_arcs, label_proj_head, label_proj_dep, s_upos, s_xpos, s_attrs = self.forward(new_seqs,
                                                                                                lang_ids=lang_ids)

            s_arcs = s_arcs.detach().cpu().numpy()
            s_lens = []
            for ii in range(len(data)):
                s_lens.append(len(new_seqs[ii]))
            pred_heads, labels = self.get_tree(s_arcs, s_lens, label_proj_head, label_proj_dep)

            s_labels = labels.detach().cpu().numpy()
            e_idx = 0
            for b_idx in range(s_upos.shape[0]):
                for w_idx in range(len(sequences[sent_id])):
                    if w_idx >= len(pred_heads[b_idx]):
                        continue
                    while sequences[sent_id][word_id].is_compound_entry:
                        sequences[sent_id][word_id].head = '_'
                        word_id += 1
                    pred_labels = np.argmax(s_labels[b_idx, e_idx])

                    if word_id < len(data[b_idx]):
                        sequences[sent_id][word_id].label = self.encodings.labels[pred_labels]
                        sequences[sent_id][word_id].head = pred_heads[b_idx][e_idx]
                    e_idx += 1
                    word_id += 1
                sent_id += 1
                word_id = 0

        return sequences


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
    tgt_arcs = []
    tgt_labels = []
    for sent in data:
        upos_int = []
        xpos_int = []
        attrs_int = []
        arc_int = []
        label_int = []
        for entry in sent:
            arc_int.append(entry.head)
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
            if entry.label in encodings.label2int:
                label_int.append(encodings.label2int[entry.label])
            else:
                label_int.append(encodings.label2int['<UNK>'])
        for _ in range(max_sent_size - len(sent)):
            upos_int.append(encodings.upos2int['<PAD>'])
            xpos_int.append(encodings.xpos2int['<PAD>'])
            attrs_int.append(encodings.attrs2int['<PAD>'])
            arc_int.append(0)
            label_int.append(0)
        tgt_upos.append(upos_int)
        tgt_xpos.append(xpos_int)
        tgt_attrs.append(attrs_int)
        tgt_arcs.append(arc_int)
        tgt_labels.append(label_int)

    import torch
    return torch.tensor(tgt_arcs, device=device), torch.tensor(tgt_labels, device=device), \
           torch.tensor(tgt_upos, device=device), torch.tensor(tgt_xpos, device=device), \
           torch.tensor(tgt_attrs, device=device)


def _eval(parser, dataset, encodings, device='cpu'):
    parser.eval()
    total = 0
    upos_ok = 0
    xpos_ok = 0
    attrs_ok = 0
    arcs_ok = 0
    labels_ok = 0
    num_batches = len(dataset.sequences) // params.batch_size
    if len(dataset.sequences) % params.batch_size != 0:
        num_batches += 1
    total_words = 0
    import tqdm
    pgb = tqdm.tqdm(range(num_batches), desc='\tEvaluating', ncols=80)
    parser.eval()
    for batch_idx in pgb:
        start = batch_idx * params.batch_size
        stop = min(len(dataset.sequences), start + params.batch_size)
        data = []
        lang_ids = []
        for ii in range(stop - start):
            data.append(dataset.sequences[start + ii][0])
            total_words += len(dataset.sequences[start + ii][0])
            lang_ids.append(dataset.sequences[start + ii][1])
        with torch.no_grad():
            s_arcs, label_proj_head, label_proj_dep, s_upos, s_xpos, s_attrs = parser(data, lang_ids=lang_ids)

        tgt_arcs, tgt_labels, tgt_upos, tgt_xpos, tgt_attrs = _get_tgt_labels(data, encodings, device=device)
        s_arcs = s_arcs.detach().cpu().numpy()
        s_lens = []
        for ii in range(len(data)):
            s_lens.append(len(data[ii]))
        pred_heads, labels = parser.get_tree(s_arcs, s_lens, label_proj_head, label_proj_dep)
        s_upos = s_upos.detach().cpu().numpy()
        s_xpos = s_xpos.detach().cpu().numpy()
        s_attrs = s_attrs.detach().cpu().numpy()
        tgt_upos = tgt_upos.detach().cpu().numpy()
        tgt_xpos = tgt_xpos.detach().cpu().numpy()
        tgt_attrs = tgt_attrs.detach().cpu().numpy()
        s_labels = labels.detach().cpu().numpy()
        for b_idx in range(tgt_upos.shape[0]):
            for w_idx in range(tgt_upos.shape[1]):
                # np.argmax(s_arcs[b_idx, w_idx])
                pred_upos = np.argmax(s_upos[b_idx, w_idx])
                pred_xpos = np.argmax(s_xpos[b_idx, w_idx])
                pred_attrs = np.argmax(s_attrs[b_idx, w_idx])
                pred_labels = np.argmax(s_labels[b_idx, w_idx])

                if tgt_upos[b_idx, w_idx] != 0:
                    if w_idx >= len(pred_heads[b_idx]):
                        print(b_idx, w_idx)
                        print("\n\n")
                        print(pred_heads)
                        print(data[b_idx])

                    pred_arc = pred_heads[b_idx][w_idx]
                    total += 1
                    if pred_upos == tgt_upos[b_idx, w_idx]:
                        upos_ok += 1
                    if pred_xpos == tgt_xpos[b_idx, w_idx]:
                        xpos_ok += 1
                    if pred_attrs == tgt_attrs[b_idx, w_idx]:
                        attrs_ok += 1
                    if pred_arc == tgt_arcs[b_idx, w_idx]:
                        arcs_ok += 1
                    if pred_arc == tgt_arcs[b_idx, w_idx] and pred_labels == tgt_labels[b_idx, w_idx]:
                        labels_ok += 1

    return arcs_ok / total, labels_ok / total, upos_ok / total, xpos_ok / total, attrs_ok / total


def _parse(parser, dataset, encodings, device='cpu'):
    parser.eval()
    num_batches = len(dataset.sequences) // params.batch_size
    if len(dataset.sequences) % params.batch_size != 0:
        num_batches += 1
    total_words = 0
    import tqdm
    pgb = tqdm.tqdm(range(num_batches), desc='\tEvaluating', ncols=80)
    parser.eval()
    import copy
    new_dataset = copy.deepcopy(dataset)
    sent_id = 0
    word_id = 0
    for batch_idx in pgb:
        start = batch_idx * params.batch_size
        stop = min(len(dataset.sequences), start + params.batch_size)
        data = []
        lang_ids = []
        for ii in range(stop - start):
            data.append(dataset.sequences[start + ii][0])
            total_words += len(dataset.sequences[start + ii][0])
            lang_ids.append(dataset.sequences[start + ii][1])
        with torch.no_grad():
            s_arcs, label_proj_head, label_proj_dep, s_upos, s_xpos, s_attrs = parser(data, lang_ids=lang_ids)

        s_arcs = s_arcs.detach().cpu().numpy()
        s_lens = []
        for ii in range(len(data)):
            s_lens.append(len(data[ii]))
        pred_heads, labels = parser.get_tree(s_arcs, s_lens, label_proj_head, label_proj_dep)
        s_upos = s_upos.detach().cpu().numpy()
        s_xpos = s_xpos.detach().cpu().numpy()
        s_attrs = s_attrs.detach().cpu().numpy()
        s_labels = labels.detach().cpu().numpy()
        for b_idx in range(s_upos.shape[0]):
            for w_idx in range(len(pred_heads[b_idx])):
                # np.argmax(s_arcs[b_idx, w_idx])
                pred_upos = np.argmax(s_upos[b_idx, w_idx])
                pred_xpos = np.argmax(s_xpos[b_idx, w_idx])
                pred_attrs = np.argmax(s_attrs[b_idx, w_idx])
                pred_labels = np.argmax(s_labels[b_idx, w_idx])

                if word_id < len(data[b_idx]):
                    new_dataset.sequences[sent_id][0][word_id].upos = encodings.upos_list[pred_upos]
                    new_dataset.sequences[sent_id][0][word_id].xpos = encodings.xpos_list[pred_xpos]
                    new_dataset.sequences[sent_id][0][word_id].attrs = encodings.attrs_list[pred_attrs]
                    new_dataset.sequences[sent_id][0][word_id].label = encodings.labels[pred_labels]
                    new_dataset.sequences[sent_id][0][word_id].head = pred_heads[b_idx][w_idx]
                    word_id += 1
            sent_id += 1
            word_id = 0

    return new_dataset


def _start_train(params, trainset, devset, encodings, parser, criterion, trainer):
    patience_left = params.patience
    epoch = 1

    best_arc = 0
    best_label = 0
    encodings.save('{0}.encodings'.format(params.store))
    parser.config.num_languages = parser.num_languages
    parser.config.save('{0}.conf'.format(params.store))
    # _eval(parser, devset, encodings, device=params.device)
    criterionNLL = criterion[1]
    criterion = criterion[0]
    while patience_left > 0:
        if epoch > parser.config.warming_epochs:
            patience_left -= 1
        sys.stdout.write('\n\nStarting epoch ' + str(epoch) + '\n')
        sys.stdout.flush()
        random.shuffle(trainset.sequences)
        num_batches = len(trainset.sequences) // params.batch_size
        if len(trainset.sequences) % params.batch_size != 0:
            num_batches += 1
        total_words = 0
        epoch_loss = 0
        import tqdm
        pgb = tqdm.tqdm(range(num_batches), desc='\tloss=NaN', ncols=80)
        parser.train()
        if epoch == parser.config.warming_epochs:
            print("Warming done")
        for batch_idx in pgb:
            start = batch_idx * params.batch_size
            stop = min(len(trainset.sequences), start + params.batch_size)
            data = []
            lang_ids = []
            for ii in range(stop - start):
                data.append(trainset.sequences[start + ii][0])
                lang_ids.append(trainset.sequences[start + ii][1])
                total_words += len(trainset.sequences[start + ii][0])

            s_arcs, proj_label_head, proj_label_dep, s_aux_upos, s_aux_xpos, s_aux_attrs = parser(data,
                                                                                                  lang_ids=lang_ids,
                                                                                                  warmup=epoch < parser.config.warming_epochs)
            tgt_arc, tgt_label, tgt_upos, tgt_xpos, tgt_attrs = _get_tgt_labels(data, encodings, device=params.device)

            pred_heads, pred_labels = parser.get_tree(None, None, proj_label_head, proj_label_dep, gs_heads=tgt_arc)
            # loss = nll_loss(s_arcs.reshape(-1, s_arcs.shape[-1]), tgt_arc.view(-1))
            loss_label = criterion(pred_labels.view(-1, pred_labels.shape[-1]), tgt_label.view(-1))

            if epoch < parser.config.warming_epochs:
                loss = ((criterion(s_aux_upos.view(-1, s_aux_upos.shape[-1]), tgt_upos.view(-1)) +
                         criterion(s_aux_xpos.view(-1, s_aux_xpos.shape[-1]), tgt_xpos.view(-1)) +
                         criterion(s_aux_attrs.view(-1, s_aux_attrs.shape[-1]), tgt_attrs.view(-1))) * 0.34)
            else:
                loss = (criterionNLL(s_arcs.reshape(-1, s_arcs.shape[-1]), tgt_arc.view(-1)))

            # loss_aux = criterion(s_aux_upos.view(-1, s_aux_upos.shape[-1]), tgt_upos.view(-1)) * parser.config.aux_softmax_weight

            total_loss = loss + loss_label
            trainer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parser.parameters(), 5.)
            trainer.step()
            epoch_loss += total_loss.item()
            pgb.set_description('\tloss={0:.4f}'.format(total_loss.item()))
        acc_arc, acc_label, acc_upos, acc_xpos, acc_attrs = _eval(parser, devset, encodings)
        fn = '{0}.last'.format(params.store)
        parser.save(fn)
        sys.stdout.flush()
        sys.stderr.flush()
        if best_arc < acc_arc:
            best_arc = acc_arc
            sys.stdout.write('\tStoring {0}.bestUAS\n'.format(params.store))
            sys.stdout.flush()
            fn = '{0}.bestUAS'.format(params.store)
            parser.save(fn)
            patience_left = params.patience

        if best_label < acc_label:
            best_label = acc_label
            sys.stdout.write('\tStoring {0}.bestLAS\n'.format(params.store))
            sys.stdout.flush()
            fn = '{0}.bestLAS'.format(params.store)
            parser.save(fn)
            patience_left = params.patience

        sys.stdout.write("\tAVG Epoch loss = {0:.6f}\n".format(epoch_loss / num_batches))
        sys.stdout.flush()
        sys.stdout.write(
            "\tValidation accuracy UAS={3:.4}, LAS={4:.4f}, UPOS={0:.4f}, XPOS={1:.4f}, ATTRS={2:.4f}\n".format(
                acc_upos, acc_xpos,
                acc_attrs, acc_arc, acc_label))
        sys.stdout.flush()
        epoch += 1


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
    from cube.io_utils.config import ParserConfig

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
    config = ParserConfig()
    parser = Parser(config, encodings, len(train_list), target_device=params.device)
    if params.resume:
        parser.load('{0}.last'.format(params.store))
    if params.device != 'cpu':
        parser.cuda(params.device)

    import torch.optim as optim
    import torch.nn as nn
    trainer = optim.Adam(parser.parameters(), lr=1e-4)  # lr=2e-3, amsgrad=False, weight_decay=0.01, betas=(0.9, 0.9))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterionNLL = nn.NLLLoss()
    if params.device != 'cpu':
        criterion.cuda(params.device)
    _start_train(params, trainset, devset, encodings, parser, [criterion, criterionNLL], trainer)


def do_test(params):
    num_languages = 11
    from cube.config import ParserConfig
    from cube.io_utils.conll import Dataset
    dataset = Dataset()
    dataset.load_language(params.test_file, params.lang_id)
    encodings = Encodings()
    encodings.load(params.model_base + '.encodings')
    config = ParserConfig()
    config.load(params.model_base + '.conf')
    parser = Parser(config, encodings, num_languages, target_device=params.device)
    parser.load(params.model_base + '.bestUAS')
    uas, las, upos_acc, xpos_acc, attrs_acc = _eval(parser, dataset, encodings, device=params.device)
    sys.stdout.write(
        'UAS={0}, LAS={1}, UPOS={2}, XPOS={3}, ATTRS={4}\n'.format(uas, las, upos_acc, xpos_acc, attrs_acc))


def do_parse(params):
    num_languages = 11
    from cube.config import ParserConfig
    from cube.io_utils.conll import Dataset
    dataset = Dataset()
    dataset.load_language(params.test_file, params.lang_id)
    encodings = Encodings()
    encodings.load(params.model_base + '.encodings')
    config = ParserConfig()
    config.load(params.model_base + '.conf')
    parser = Parser(config, encodings, num_languages, target_device=params.device)
    parser.load(params.model_base + '.bestUAS')
    new_dataset = _parse(parser, dataset, encodings, device=params.device)

    for seq in new_dataset.sequences:
        for entry in seq[0]:
            sys.stdout.write(str(entry))
        print()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store', dest='train_file',
                      help='Start building a parser model')
    parser.add_option('--patience', action='store', type='int', default=20, dest='patience',
                      help='Number of epochs before early stopping (default=20)')
    parser.add_option('--store', action='store', dest='store', help='Output base', default='parser')
    parser.add_option('--batch-size', action='store', type='int', default=32, dest='batch_size',
                      help='Number of epochs before early stopping (default=32)')
    parser.add_option('--device', action='store', dest='device', default='cpu',
                      help='What device to use for models: cpu, cuda:0, cuda:1 ...')
    parser.add_option('--test', action='store_true', dest='test', help='Test the traine model')
    parser.add_option('--test-file', action='store', dest='test_file')
    parser.add_option('--lang-id', action='store', dest='lang_id', type='int', default=0)
    parser.add_option('--model-base', action='store', dest='model_base')
    parser.add_option('--process', action='store_true')
    parser.add_option('--resume', action='store_true', dest='resume')

    (params, _) = parser.parse_args(sys.argv)

    if params.train_file:
        do_train(params)
    if params.test:
        do_test(params)
    if params.process:
        do_parse(params)
