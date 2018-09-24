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
import io
from cube.misc.misc import fopen

class Dataset:
    def __init__(self, file=None):
        if file is not None:
            sys.stdout.write("Reading " + file + "... ")
            sys.stdout.flush()
            with fopen(file, "r") as f:
                lines = f.readlines()
                
            self.sequences = self._make_sequences(lines)
            sys.stdout.write("found " + str(len(self.sequences)) + " sequences\n")

    def _make_sequences(self, lines):
        sequences = []
        in_sequence = False
        seq = []
        for line in lines:
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            if (not line.startswith("#") or in_sequence) and line != '':
                parts = line.split("\t")
                s = ConllEntry(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8],
                               parts[9])
                seq.append(s)
                in_sequence = True
            elif line == "":
                in_sequence = False
                if len(seq) > 0:
                    sequences.append(seq)
                seq = []

        return sequences

    def write(self, filename):
        with fopen(filename, 'w') as file:
            for sequence in self.sequences:
                for entry in sequence:
                    file.write(str(entry.index))
                    file.write("\t")
                    if isinstance(entry.word, str):
                        file.write(entry.word)
                    else:
                        file.write(entry.word.encode('utf-8'))
                    file.write("\t")
                    if isinstance(entry.lemma, str):
                        file.write(entry.lemma)
                    else:
                        file.write(entry.lemma.encode('utf-8'))
                    file.write("\t")
                    file.write(entry.upos)
                    file.write("\t")
                    file.write(entry.xpos)
                    file.write("\t")
                    file.write(entry.attrs)
                    file.write("\t")
                    file.write(str(entry.head))
                    file.write("\t")
                    file.write(entry.label)
                    file.write("\t")
                    file.write(entry.deps)
                    file.write("\t")
                    file.write(entry.space_after)
                    file.write("\n")
                file.write("\n")

    def write_stdout(self):
        import sys
        file = sys.stdout
        for sequence in self.sequences:
            for entry in sequence:
                file.write(str(entry.index))
                file.write("\t")
                if isinstance(entry.word, str):
                    file.write(entry.word)
                else:
                    file.write(entry.word.encode('utf-8'))
                file.write("\t")
                if isinstance(entry.lemma, str):
                    file.write(entry.lemma)
                else:
                    file.write(entry.lemma.encode('utf-8'))
                file.write("\t")
                file.write(entry.upos)
                file.write("\t")
                file.write(entry.xpos)
                file.write("\t")
                file.write(entry.attrs)
                file.write("\t")
                file.write(str(entry.head))
                file.write("\t")
                file.write(entry.label)
                file.write("\t")
                file.write(entry.deps)
                file.write("\t")
                file.write(entry.space_after)
                file.write("\n")
            file.write("\n")


class Encodings:
    def __init__(self):
        return ""


class ConllEntry:
    def __init__(self, index, word, lemma, upos, xpos, attrs, head, label, deps, space_after):
        self.index, self.is_compound_entry = self._int_try_parse(index)
        self.word = word
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.attrs = attrs
        self.head, _ = self._int_try_parse(head)
        self.label = label
        self.deps = deps
        self.space_after = space_after

    def _int_try_parse(self, value):
        try:
            return int(value), False
        except ValueError:
            return value, True
