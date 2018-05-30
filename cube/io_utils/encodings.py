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
import re


class Encodings(object):

    def __init__(self, verbose = True):
        self.word_list = {}
        self.hol_word_list = []
        self.char2int = {}
        self.label2int = {}
        self.labels = []
        self.word2int = {}
        self.upos2int = {}
        self.xpos2int = {}
        self.attrs2int = {}
        self.upos_list = []
        self.xpos_list = []
        self.attrs_list = []
        self.characters = []        
        self.verbose = verbose

    def compute(self, train, dev, tag_type=None, word_cutoff=2, char_cutoff=5):
        if self.verbose:
            sys.stdout.write("Computing encoding maps... ")
            sys.stdout.flush()
        self.char2int['<UNK>'] = 0
        self.characters.append("<UNK>")
        self.char2int[' '] = 1
        self.characters.append(" ")
        char_count = {}
        word_count = {}
        for seq in train.sequences:
            for entry in seq:
                word = entry.word.decode('utf-8').lower()
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] = word_count[word] + 1
                if word not in self.word_list:
                    self.word_list[word] = 0  # word is inside trainset
                uniword = unicode(entry.word, 'utf-8').lower()
                uniword = re.sub('\d', '0', uniword)
                for i in range(len(uniword)):
                    char = uniword[i].lower()
                    if char not in char_count:
                        char_count[char] = 1
                    else:
                        char_count[char] = char_count[char] + 1

                        # if char not in self.char2int:
                        #    self.char2int[char] = len(self.char2int)
                label = None
                if tag_type == 'upos':
                    label = entry.upos
                elif tag_type == 'xpos':
                    label = entry.xpos
                elif tag_type == 'attrs':
                    label = entry.attrs
                elif tag_type == 'label':
                    label = entry.label
                if label not in self.label2int:
                    self.label2int[label] = len(self.label2int)
                    self.labels.append(label)

                # morphological encodings
                if entry.upos not in self.upos2int:
                    self.upos2int[entry.upos] = len(self.upos2int)
                    self.upos_list.append(entry.upos)
                if entry.xpos not in self.xpos2int:
                    self.xpos2int[entry.xpos] = len(self.xpos2int)
                    self.xpos_list.append(entry.xpos)
                if entry.attrs not in self.attrs2int:
                    self.attrs2int[entry.attrs] = len(self.attrs2int)
                    self.attrs_list.append(entry.attrs)

        for seq in dev.sequences:
            for entry in seq:
                word = entry.word.decode('utf-8').lower()
                if word not in self.word_list:
                    self.word_list[word] = 1  # word is inside devset only

        self.word2int['<UNK>'] = 0
        self.hol_word_list.append('<UNK>')
        for word in word_count:
            if word_count[word] >= word_cutoff:
                self.word2int[word] = len(self.word2int)
                self.hol_word_list.append(word)
        for char in char_count:
            if char_count[char] >= char_cutoff:
                self.char2int[char] = len(self.char2int)
                self.characters.append(char)

        # force add digits
        for digit in xrange(10):
            ds = str(digit)
            if ds not in self.char2int:
                self.char2int[ds] = len(self.char2int)
                self.characters.append(ds)
        if self.verbose:
            sys.stdout.write("done\n")

            print "Unique words: " + str(len(self.word_list))
            print "Unique chars: " + str(len(self.char2int))
            print "Unique labels: " + str(len(self.label2int))
            print "Unique UPOS: " + str(len(self.upos2int))
            print "Unique XPOS: " + str(len(self.xpos2int))
            print "Unique ATTRS: " + str(len(self.attrs2int))
            print "Holistic word count: " + str(len(self.word2int))

    def update_wordlist(self, dataset):
        for seq in dataset.sequences:
            for entry in seq:
                word = entry.word.decode('utf-8').lower()
                if word not in self.word_list:
                    self.word_list[word] = 2  # word is inside an auxiliarly set (probably test)

    def load(self, filename):
        # We only read character2int, labels, holistic words and label2int here. word_list should be recomputed for every dataset (if deemed necessary)

        with open(filename) as f:
            line = f.readline()

            num_labels = int(line.split(" ")[1])
            if self.verbose:
                print "Loading labels " + str(num_labels)
            self.labels = [""] * num_labels
            for _ in xrange(num_labels):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.label2int[key] = value
                self.labels[value] = key

            line = f.readline()
            num_characters = int(line.split(" ")[1])
            self.characters = [""] * num_characters
            if self.verbose:
                print "Loading characters " + str(num_characters)
            for _ in xrange(num_characters):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0].decode('utf-8')
                value = int(parts[1])
                self.char2int[key] = value
                self.characters[value] = key
            line = f.readline()
            num_words = int(line.split(" ")[1])
            if self.verbose:
                print "Loading words " + str(num_words)
            for _x in xrange(num_words):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0].decode('utf-8')
                value = int(parts[1])
                self.word2int[key] = value

            # morphological attributes
            line = f.readline()
            num_labels = int(line.split(" ")[1])
            if self.verbose:
                print "Loading upos " + str(num_labels)
            self.upos_list = [""] * num_labels
            for _ in xrange(num_labels):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.upos2int[key] = value
                self.upos_list[value] = key

            line = f.readline()
            num_labels = int(line.split(" ")[1])
            self.xpos_list = [""] * num_labels
            if self.verbose:
                print "Loading xpos " + str(num_labels)
            for _ in xrange(num_labels):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.xpos2int[key] = value
                self.xpos_list[value] = key

            line = f.readline()
            num_labels = int(line.split(" ")[1])
            self.attrs_list = [""] * num_labels
            if self.verbose:
                print "Loading attrs " + str(num_labels)
            for _ in xrange(num_labels):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.attrs2int[key] = value
                self.attrs_list[value] = key
            f.close()

    def save(self, filename):
        with open(filename, "w") as f:
            f.write("LABELS " + str(len(self.label2int)) + "\n")
            for label in self.label2int:
                f.write(str(label) + "\t" + str(self.label2int[label]) + "\n")
            f.write("CHARACTERS " + str(len(self.char2int)) + "\n")
            for character in self.char2int:
                f.write(character.encode('utf-8') + "\t" + str(self.char2int[character]) + "\n")
            f.write("WORDS " + str(len(self.word2int)) + "\n")
            for word in self.word2int:
                f.write(word.encode('utf-8') + "\t" + str(self.word2int[word]) + "\n")

            f.write("UPOS " + str(len(self.upos2int)) + "\n")
            for label in self.upos2int:
                f.write(label + "\t" + str(self.upos2int[label]) + "\n")
            f.write("XPOS " + str(len(self.xpos2int)) + "\n")
            for label in self.xpos2int:
                f.write(label + "\t" + str(self.xpos2int[label]) + "\n")
            f.write("ATTRS " + str(len(self.attrs2int)) + "\n")
            for label in self.attrs2int:
                f.write(label + "\t" + str(self.attrs2int[label]) + "\n")
            f.close()
