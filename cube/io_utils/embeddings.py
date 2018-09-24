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
from scipy import spatial


class WordEmbeddings:
    def __init__(self, verbose=True):
        self.word2vec = {}
        self.word2ofs = {}
        self.word_embeddings_size = 0
        self.num_embeddings = 0
        self.cache_only = False
        self.file_pointer = None
        self.verbose = verbose

    def read_from_file(self, word_embeddings_file, word_list, full_load=False):
        self.word2vec = {}
        self.num_embeddings = 0
        if word_list is None and not full_load:
            self.cache_only = True
        f = fopen(word_embeddings_file, "r")
        first_line = True
        while True:
            ofs = f.tell()
            line = f.readline()
            if line == '':
                break
                # print ofs
            line = line.replace("\n", "").replace("\r", "")
            if first_line:
                first_line = False
            else:
                self.num_embeddings += 1
                if self.verbose:
                    if self.num_embeddings % 10000 == 0:
                        sys.stdout.write(
                            "  Scanned " + str(self.num_embeddings) + " word embeddings and added " + str(
                                len(self.word2vec)) + "  \n")
                parts = line.split(" ")
                if sys.version_info[0] == 2:
                    word = parts[0].decode('utf-8')
                else:
                    word = parts[0]
                if self.cache_only:
                    self.word2ofs[word] = ofs
                elif full_load or word in word_list:
                    embeddings = [float(0)] * (len(parts) - 2)

                    for zz in range(len(parts) - 2):
                        embeddings[zz] = float(parts[zz + 1])
                    self.word2vec[word] = embeddings
                self.word_embeddings_size = len(parts) - 2
        f.close()            
        if self.cache_only:            
            self.file_pointer = fopen(word_embeddings_file, "r")
            
            
    def get_word_embeddings(self, word):
        word = word.lower()
        if self.cache_only:
            if word in self.word2ofs:
                self.file_pointer.seek(self.word2ofs[word])
                line = self.file_pointer.readline()
                parts = line.split(" ")
                embeddings = [float(0)] * (len(parts) - 2)
                for zz in range(len(parts) - 2):
                    embeddings[zz] = float(parts[zz + 1])
                return embeddings, True
            else:
                return None, False

        elif word in self.word2vec:
            return self.word2vec[word], True
        else:
            return None, False

    def get_closest_word(self, vector):
        best_distance = -1.0
        best_word = '<UNK>'
        for word in self.word2vec:
            similarity = 1.0 - spatial.distance.cosine(vector, self.word2vec[word])
            if similarity > best_distance:
                best_distance = similarity
                best_word = word
        return best_word
