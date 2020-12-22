#
# Author: Tiberiu Boros
#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
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

import sys
from cube.misc.misc import fopen

class MTDataset:
    def __init__(self, src, dst):
        sys.stdout.write("Reading files '"+src+"' and '"+dst+"'")
        sys.stdout.flush()
        source_lines=fopen(src, "r").readlines()
        destination_lines=fopen(dst,"r").readlines()
        self.sequences=self._make_sequences(source_lines, destination_lines)
        sys.stdout.write(" found "+str(len(self.sequences))+" pairs\n")

    def _make_sequences(self, src_lines, dst_lines):
        import re
        sequences=[]
        if len(src_lines)!=len(dst_lines):
            sys.stdout.write(" WARNING: datasets do not match")
        for src, dst in zip (src_lines, dst_lines):
            tok_src=src.split(" ")
            tok_dst=dst.split(" ")
            from conll import ConllEntry
            conll_src=[]
            conll_dst=[]
            for index, tok in zip(range(len(tok_src)), tok_src):
                if tok.decode('utf-8').lower()!=tok.decode('utf-8'):
                    tok='<PROPN>'
                tok=re.sub('\d','0',tok)
                if tok!='\n':
                    conll_src.append(ConllEntry(index, tok, "", "", "", "", 0, "", "", ""))

            for index, tok in zip(range(len(tok_dst)), tok_dst):
                if tok.decode('utf-8').lower()!=tok.decode('utf-8'):
                    tok='<PROPN>'
                tok = re.sub('\d', '0', tok)
                if tok != '\n':
                    conll_dst.append(ConllEntry(index, tok, "", "", "", "", 0, "", "", ""))

            sequences.append(MTSequence(conll_src, conll_dst))
        return sequences

    def to_conll_dataset(self, type='src'):
        conllDataset = FakeCONLLDataset()
        for mt_seq in self.sequences:
            if type == 'src':
                conllDataset.sequences.append(mt_seq.src)
            else:
                conllDataset.sequences.append(mt_seq.dst)
        return conllDataset

class FakeCONLLDataset():
    def __init__(self):
        self.sequences=[]


class MTSequence:
    def __init__(self, conll_src, conll_dst):
        self.src=conll_src
        self.dst=conll_dst
