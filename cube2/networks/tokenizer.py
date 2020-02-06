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

import torch
import torch.nn as nn
from cube2.networks.modules import LinearNorm


class SentenceSplitter(nn.Module):
    def __init__(self, config, encodings):
        super(SentenceSplitter, self).__init__()
        self.config = config
        self.encodings = encodings

        self.char_lookup = nn.Embedding(len(encodings.char2int), config.char_emb_size, padding_idx=0)
        input_emb_size = config.char_emb_size
        if config.num_languages != 1:
            self.lang_lookup = nn.Embedding(config.num_languages, config.lang_emb_size, padding_idx=0)
            input_emb_size += config.lang_emb_size

        conv_list = [nn.Conv1d(input_emb_size,
                               self.config.ss_conv_filters,
                               self.config.ss_conv_kernel,
                               padding=self.config.ss_conv_kernel // 2)]
        for _ in range(self.config.ss_conv_layers - 1):
            conv_list.append(nn.Conv1d(self.config.ss_conv_filters,
                                       self.config.ss_conv_filters,
                                       self.config.ss_conv_kernel,
                                       padding=self.config.ss_conv_kernel // 2))

        self.conv = nn.ModuleList(conv_list)

        self.output = LinearNorm(self.config.ss_conv_filters, 3)

    def forward(self, char_idx, lang_idx=None):
        char_emb = self.char_lookup(char_idx)
        if lang_idx is not None:
            lang_emb = self.lang_lookup(lang_idx)
            input_emb = torch.cat((char_emb, lang_emb), dim=-1)
        else:
            input_emb = char_emb

        input_emb = input_emb.permute(0, 2, 1)
        hidden = input_emb
        res = None
        for conv in self.conv:
            c_out = conv(hidden)
            if res is not None:
                hidden = c_out + res
            else:
                hidden = c_out
            res = c_out
            hidden = torch.dropout(torch.relu(hidden), 0.5, self.training)

        hidden = hidden.permute(0, 2, 1)
        output = self.output(hidden)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
