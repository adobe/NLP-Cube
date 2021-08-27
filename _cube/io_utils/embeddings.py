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

    def load_dummy_embeddings(self):
        self.word_embeddings_size = 300
        self.num_embeddings = 1
        self.cache_only = False
        self.word2vec["</s>"] = [-0.00291,-0.18625,0.058028,-0.29661,0.068632,0.28946,0.0051565,0.17799,0.13018,-0.19517,-0.19083,0.0064972,-0.10813,-0.24357,-0.37569,-0.073037,0.16525,0.096274,-0.21967,-0.060706,-0.10286,-0.084604,0.1744,-0.031249,-0.10277,0.042256,0.20297,-0.039783,-0.028972,0.19946,-0.0046892,0.28466,-0.069435,0.068676,-0.023165,-0.046733,0.018789,-0.22607,-0.21216,0.17036,0.17399,0.066895,0.10422,0.15653,0.18378,-0.11769,0.1509,-0.081692,0.23579,-0.0093485,0.15167,-0.0097952,-0.035584,-0.0023132,0.10254,0.10207,-0.28487,-0.14675,-0.073192,0.056664,-0.019519,0.088953,0.079022,0.022475,-0.27715,0.08987,-0.038999,0.0028215,0.096766,0.32482,-0.18077,-0.15867,0.042736,-0.098426,0.19944,-0.31784,-0.1702,0.069564,-0.13843,-0.002799,0.3482,-0.011427,0.063471,-0.067671,0.013669,0.29324,0.089274,0.21365,0.27761,-0.31488,0.093548,0.020789,-0.020138,0.1133,0.14776,0.42006,-0.29999,-0.051176,0.11858,0.032608,0.33327,-0.19025,0.059251,-0.033972,-0.11209,0.11292,0.042999,0.23898,-0.056097,-0.052971,0.22852,0.050305,-0.012704,-0.081334,-0.027748,0.3725,-0.13769,0.17957,-0.094775,0.029658,-0.025596,0.32383,-0.039333,-0.2727,-0.29954,0.045625,0.11779,-0.29941,0.15999,-0.068357,0.083792,0.1545,-0.077238,0.15015,0.016603,-0.035233,0.16562,-0.19231,-0.37249,0.083253,-0.1229,-0.12096,0.33754,0.29348,0.21091,-0.3435,0.13705,-0.065175,-0.29146,-0.041733,-0.28036,0.16005,0.0086172,-0.020325,0.012445,-0.15517,0.20095,-0.0010922,0.23908,0.27801,0.13009,-0.1019,-0.059306,0.15052,-0.049257,0.10735,0.24883,-0.035315,-0.015704,0.1297,-0.26409,-0.17914,-0.34641,0.19633,0.12695,0.20356,0.32595,-0.013281,0.068872,-0.063561,-0.076011,0.066515,-0.18736,-0.058394,-0.11234,0.17197,0.24167,0.11705,0.012847,0.040238,0.092364,0.33407,-0.1206,-0.074965,0.22935,-0.035572,0.10584,-0.097787,-0.063045,-0.13527,0.053755,-0.33137,-0.051164,0.02706,0.059661,-0.32057,0.3829,0.1358,-0.086782,0.11528,-0.23391,0.21434,-0.12766,0.059699,-0.25511,-0.039314,-0.12894,-0.012722,0.2139,0.10244,-0.21011,-0.21161,-0.012924,0.19177,0.04161,0.084953,-0.06967,0.066996,-0.058172,0.25607,-0.2864,0.0041426,-0.38308,-0.021462,-0.17859,-0.32166,-0.029291,0.11121,0.18469,0.16992,-0.015047,-0.2933,-0.28637,0.2433,-0.042533,0.20242,-0.1547,-0.31574,-0.13264,0.11957,-0.37728,0.019524,-0.2068,0.083229,-0.12357,0.048097,-0.41851,0.55805,-0.024595,-0.15514,-0.0063529,0.21332,0.33929,0.32646,-0.079572,0.34776,-0.077371,0.13704,-0.38439,-0.0092563,0.055568,0.048388,0.081304,0.17917,0.054183,-0.072919,0.12038,-0.14533,0.25047,0.32786,0.31443,-0.16633,0.19101,0.069506,0.27939,-0.078308,0.1836,-0.07276,-0.057231,0.017672,-0.083664,0.11733,-0.0621,-0.0044557,-0.31261,0.069088,0.13869,-0.072683,-0.11379,0.037591]
        
    def read_from_file(self, word_embeddings_file, word_list, full_load=False):
        self.word2vec = {}
        self.num_embeddings = 0
        if word_list is None and not full_load:
            self.cache_only = True
        with fopen(word_embeddings_file, "r") as f:
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
