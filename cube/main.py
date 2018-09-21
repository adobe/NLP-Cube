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

import dynet_config
import optparse
import sys
import os
import copy
from misc.misc import fopen

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store', dest='train',
                      choices=['tagger', 'parser', 'lemmatizer', 'tokenizer', 'mt', 'compound', 'ner'],
                      help='select which model to train: tagger, parser, lemmatizer, tokenizer, mt, compound, ner')
    parser.add_option('--train-file', action='store', dest='train_file', help='location of the train dataset')
    parser.add_option('--dev-file', action='store', dest='dev_file', help='location of the dev dataset')
    parser.add_option('--embeddings', action='store', dest='embeddings',
                      help='location of the pre-computed word embeddings file')
    parser.add_option('--patience', action='store', dest='itters', help='no improvement stopping condition',
                      default='20', type='int')
    parser.add_option('--store', action='store', dest='output_base', help='output base for model location')
    parser.add_option('--aux-softmax', action='store', dest='aux_softmax_weight',
                      help='weight for the auxiliarly softmax', default='0.2',
                      type='float')  # should remove, it is a parser specific config
    parser.add_option("--config", action='store', dest='config', help='configuration file to load')
    parser.add_option("--test", action='store', dest='test', choices=['tagger', 'parser', 'lemmatizer', 'tokenizer'],
                      help='select what model to test')
    parser.add_option("--model", action='store', dest='model', help='location where model is stored')
    parser.add_option("--raw-train-file", action='store', dest='raw_train_file', help='location of the raw train file')
    parser.add_option("--raw-dev-file", action='store', dest='raw_dev_file', help='location of the raw dev file')
    parser.add_option("--raw-test-file", action='store', dest='raw_test_file', help='location of the raw test file')
    parser.add_option("--test-file", action='store', dest='test_file', help='location of the test dataset')
    parser.add_option("--run", action='store', dest='run',
                      help='Pipeline components: [all, tokenizer, compound, lemmatizer, tagger, parser]')
    parser.add_option("--models", action='store', dest='models', help='Folder path to where the components are stored.')

    parser.add_option("--batch-size", action='store', dest='batch_size', default='10', type='int',
                      help='number of sentences in a single batch (default=10)')
    parser.add_option("--set-mem", action='store', dest='memory', default='2048', type='int',
                      help='preallocate memory for batch training (default 2048)')
    parser.add_option("--autobatch", action='store_true', dest='autobatch',
                      help='turn on/off dynet autobatching')
    parser.add_option("--use-gpu", action='store_true', dest='gpu',
                      help='turn on/off GPU support')
    parser.add_option("--decay", action='store', dest='decay', default=0, type='float',
                      help='set value for weight decay regularization')
    parser.add_option("--params", action='store', dest='params', type='str', help='external params')

    parser.add_option("--input-file", action='store', dest='input_file',
                      help='path to the input file (either text or conllu format)')
    parser.add_option("--output-file", action='store', dest='output_file', help='path to output file to generate')
    parser.add_option("--parser-decoder", action='store', dest='decoder', default='greedy', choices=['greedy', 'mst'],
                      help='what algorithm to use for parser decoding at runtime (greedy or mst)')  # should remove, it is a parser specific config
    parser.add_option("--model-file", action='store', dest='model_base', help='what model to use')
    parser.add_option("--mt-train-src", action='store', dest='mt_train_src',
                      help='train file for source language')
    parser.add_option("--mt-train-dst", action='store', dest='mt_train_dst',
                      help='train file for destination language')
    parser.add_option("--mt-dev-src", action='store', dest='mt_dev_src',
                      help='dev file for source language')
    parser.add_option("--mt-dev-dst", action='store', dest='mt_dev_dst',
                      help='dev file for destination language')
    parser.add_option("--mt-test-src", action='store', dest='mt_test_src',
                      help='test file for source language')
    parser.add_option("--mt-test-dst", action='store', dest='mt_test_dst',
                      help='test file for destination language')
    parser.add_option("--mt-source-embeddings", action='store', dest='mt_source_embeddings',
                      help='embeddings file for sourcelanguage')
    parser.add_option("--mt-destination-embeddings", action='store', dest='mt_destination_embeddings',
                      help='embeddings file for destination language')

    parser.add_option('--start-server', action='store_true', dest='server',
                      help='start an embedded webserver')
    parser.add_option('--server-port', action='store', dest='port', type='int', default='80',
                      help='set the runtime server port')
    parser.add_option('--model-lemmatization', action='store', dest='model_lemmatization',
                      help='precomputed lemmatization model')
    parser.add_option('--model-tokenization', action='store', dest='model_tokenization',
                      help='precomputed tokenization model')
    parser.add_option('--model-tagging', action='store', dest='model_tagging', help='precomputed tagging model')
    parser.add_option('--model-parsing', action='store', dest='model_parsing', help='precomputed parsing model')

    (params, _) = parser.parse_args(sys.argv)

    memory = int(params.memory)
    if params.autobatch:
        autobatch = True
    else:
        autobatch = False
    dynet_config.set(mem=memory, random_seed=9, autobatch=autobatch, weight_decay=params.decay)
    if params.gpu:
        dynet_config.set_gpu()

from io_utils.conll import Dataset
from io_utils.cupt import CUPTDataset
from io_utils.mt import MTDataset
from io_utils.config import TokenizerConfig
from io_utils.config import TaggerConfig
from io_utils.config import ParserConfig
from io_utils.config import LemmatizerConfig
from io_utils.config import NMTConfig
from io_utils.config import TieredTokenizerConfig
from io_utils.config import CompoundWordConfig
from io_utils.config import GDBConfig
from io_utils.embeddings import WordEmbeddings
from io_utils.encodings import Encodings
from io_utils.trainers import TokenizerTrainer
from io_utils.trainers import TaggerTrainer
from io_utils.trainers import ParserTrainer
from io_utils.trainers import LemmatizerTrainer
from io_utils.trainers import MTTrainer
from io_utils.trainers import CompoundWordTrainer
from io_utils.trainers import NERTrainer
from generic_networks.tokenizers import BDRNNTokenizer
from generic_networks.taggers import BDRNNTagger
from generic_networks.parsers import BDRNNParser
from generic_networks.lemmatizers import FSTLemmatizer
from generic_networks.translators import BRNNMT
from generic_networks.tokenizers import TieredTokenizer
from generic_networks.token_expanders import CompoundWordExpander
from generic_networks.ner import GDBNer


def parse_test(params):
    if params.test == "parser":
        print ("Running " + params.test)
        print ("==PARAMETERS==")
        print ("EMBEDDINGS: " + params.embeddings)
        print ("MODEL FILE: " + params.model_base)
        print ("DECODER: " + params.decoder)
        print ("OUTPUT: " + params.output_file)
        print ("CONFIG FILE: " + str(params.config))
        print ("==============\n")

        testset = Dataset(params.test_file)
        encodings = Encodings()
        encodings.load(params.model_base + ".encodings")
        encodings.update_wordlist(testset)
        print ("Updated word list: " + str(len(encodings.word_list)))
        config = ParserConfig(filename=params.config)
        embeddings = WordEmbeddings()
        embeddings.read_from_file(params.embeddings, encodings.word_list)
        parser = BDRNNParser(config, encodings, embeddings)
        parser.load(params.model_base + ".bestUAS")
        if params.decoder == 'mst':
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!USING MST DECODER")
            from graph.decoders import MSTDecoder
            parser.decoder = MSTDecoder()
        f = fopen(params.output_file, "w")
        last_proc = 0
        index = 0
        for seq in testset.sequences:
            index += 1
            proc = index * 100 / len(testset.sequences)
            if proc % 5 == 0 and proc != last_proc:
                last_proc = proc
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()

            rez = parser.tag(seq)
            iSeq = 0
            iRez = 0
            while iSeq < len(seq):
                while seq[iSeq].is_compound_entry:
                    iSeq += 1
                seq[iSeq].xpos = rez[iRez].xpos
                seq[iSeq].upos = rez[iRez].upos
                seq[iSeq].attrs = rez[iRez].attrs
                seq[iSeq].head = rez[iRez].head
                seq[iSeq].label = rez[iRez].label
                seq[iSeq].lemma = rez[iRez].lemma
                iSeq += 1
                iRez += 1

            for entry in seq:
                f.write(str(entry.index) + "\t" + str(entry.word) + "\t" + str(entry.lemma) + "\t" + str(
                    entry.upos) + "\t" + str(entry.xpos) + "\t" + str(entry.attrs) + "\t" + str(
                    entry.head) + "\t" + str(entry.label) + "\t" + str(entry.deps) + "\t" + str(
                    entry.space_after) + "\n")
            f.write("\n")

        f.close()
        sys.stdout.write("\n")


def parse_train(params):
    if params.train == 'mt':
        print ("Starting training for " + params.train)
        print ("==PARAMETERS==")
        print ("SRC TRAIN FILE: " + params.mt_train_src)
        print ("SRC DEV FILE: " + params.mt_dev_src)
        print ("SRC TEST FILE: " + str(params.mt_test_src))
        print ("SRC EMBEDDINGS FILE: " + params.mt_source_embeddings)
        print ("DST TRAIN FILE: " + params.mt_train_dst)
        print ("DST DEV FILE: " + params.mt_dev_dst)
        print ("DST TEST FILE: " + str(params.mt_test_dst))
        print ("DST EMBEDDINGS FILE: " + params.mt_destination_embeddings)
        print ("STOPPING CONDITION: " + str(params.itters))
        print ("OUTPUT BASE: " + params.output_base)
        print ("AUX SOFTMAX WEIGHT: " + str(params.aux_softmax_weight))
        print ("CONFIG FILE: " + str(params.config))
        print ("==============\n")
        trainset = MTDataset(params.mt_train_src, params.mt_train_dst)
        devset = MTDataset(params.mt_dev_src, params.mt_dev_dst)
        if params.mt_test_src and params.mt_test_dst:
            testset = MTDataset(params.mt_test_src, params.mt_test_dst)
        else:
            testset = None

        config = NMTConfig(params.config)
        sys.stdout.write("--SOURCE--\n")
        sys.stdout.flush()
        src_enc = Encodings()
        src_enc.compute(trainset.to_conll_dataset('src'), devset.to_conll_dataset('src'), word_cutoff=5)
        sys.stdout.write("--DESTINATION--\n")
        sys.stdout.flush()
        dst_enc = Encodings()
        dst_enc.compute(trainset.to_conll_dataset('dst'), devset.to_conll_dataset('dst'), word_cutoff=5)
        sys.stdout.write("Reading source embeddings\n")
        src_we = WordEmbeddings()
        src_we.read_from_file(params.mt_source_embeddings, 'label', full_load=False)
        sys.stdout.write("Reading destination embeddings\n")
        dst_we = WordEmbeddings()
        dst_we.read_from_file(params.mt_destination_embeddings, 'label', full_load=False)
        nmt = BRNNMT(src_we, dst_we, src_enc, dst_enc, config)
        trainer = MTTrainer(nmt, src_enc, dst_enc, src_we, dst_we, params.itters, trainset, devset, testset=testset)
        trainer.start_training(params.output_base, batch_size=params.batch_size)

    if params.train == "tagger":
        print ("Starting training for " + params.train)
        print ("==PARAMETERS==")
        print ("TRAIN FILE: " + params.train_file)
        print ("DEV FILE: " + params.dev_file)
        if params.test_file is not None:
            print ("TEST FILE: " + params.test_file)
        print ("EMBEDDINGS FILE: " + params.embeddings)
        print ("STOPPING CONDITION: " + str(params.itters))
        print ("OUTPUT BASE: " + params.output_base)
        print ("AUX SOFTMAX WEIGHT: " + str(params.aux_softmax_weight))
        print ("CONFIG FILE: " + str(params.config))
        print ("==============\n")

        trainset = Dataset(params.train_file)
        devset = Dataset(params.dev_file)
        if params.test_file:
            testset = Dataset(params.test_file)
        else:
            testset = None
        config = TaggerConfig(params.config)
        if not config._valid:
            return

        encodings = Encodings()
        encodings.compute(trainset, devset, 'label')
        # update wordlist if testset was provided
        if params.test_file:
            encodings.update_wordlist(testset)
        embeddings = WordEmbeddings()
        embeddings.read_from_file(params.embeddings, encodings.word_list)
        tagger = BDRNNTagger(config, encodings, embeddings, aux_softmax_weight=params.aux_softmax_weight)
        trainer = TaggerTrainer(tagger, encodings, params.itters, trainset, devset, testset)
        trainer.start_training(params.output_base, batch_size=params.batch_size)

    elif params.train == "parser":
        print ("Starting training for " + params.train)
        print ("==PARAMETERS==")
        print ("TRAIN FILE: " + params.train_file)
        print ("DEV FILE: " + params.dev_file)
        if params.test_file is not None:
            print ("TEST FILE: " + params.test_file)
        print ("EMBEDDINGS FILE: " + params.embeddings)
        print ("STOPPING CONDITION: " + str(params.itters))
        print ("OUTPUT BASE: " + params.output_base)
        print ("AUX SOFTMAX WEIGHT: " + str(params.aux_softmax_weight))
        print ("CONFIG FILE: " + str(params.config))
        print ("==============\n")

        trainset = Dataset(params.train_file)
        devset = Dataset(params.dev_file)
        if params.test_file:
            testset = Dataset(params.test_file)
        else:
            testset = None
        config = ParserConfig(params.config)
        if not config._valid:
            return
        # PARAM INJECTION 
        if params.params != None:
            parts = params.params.split(":")
            for param in parts:
                variable = param.split("=")[0]
                value = param[len(variable) + 1:]
                print("External param injection: " + variable + "=" + value)
                exec ("config.__dict__[\"" + variable + "\"] = " + value)
                # END INJECTION
        encodings = Encodings()
        encodings.compute(trainset, devset, 'label')
        # update wordlist if testset was provided
        if params.test_file:
            encodings.update_wordlist(testset)
        embeddings = WordEmbeddings()
        embeddings.read_from_file(params.embeddings, encodings.word_list)
        parser = BDRNNParser(config, encodings, embeddings, aux_softmax_weight=params.aux_softmax_weight)
        trainer = ParserTrainer(parser, encodings, params.itters, trainset, devset, testset)
        trainer.start_training(params.output_base, params.batch_size)

    elif params.train == "lemmatizer":
        print("Starting training for " + params.train)
        print("==PARAMETERS==")
        print("TRAIN FILE: " + params.train_file)
        print("DEV FILE: " + params.dev_file)
        if params.test_file is not None:
            print("TEST FILE: " + params.test_file)
        print("EMBEDDINGS FILE: " + params.embeddings)
        print("STOPPING CONDITION: " + str(params.itters))
        print("OUTPUT BASE: " + params.output_base)
        print("AUX SOFTMAX WEIGHT: " + str(params.aux_softmax_weight))
        print("CONFIG FILE: " + str(params.config))
        print("==============\n")

        trainset = Dataset(params.train_file)
        devset = Dataset(params.dev_file)
        if params.test_file:
            testset = Dataset(params.test_file)
        else:
            testset = None
        config = LemmatizerConfig(params.config)
        encodings = Encodings()
        encodings.compute(trainset, devset, 'label')
        # update wordlist if testset was provided
        if params.test_file:
            encodings.update_wordlist(testset)

        embeddings = None
        lemmatizer = FSTLemmatizer(config, encodings, embeddings)
        # from generic_networks.lemmatizers import BDRNNLemmatizer
        # lemmatizer = BDRNNLemmatizer(config, encodings, embeddings)
        trainer = LemmatizerTrainer(lemmatizer, encodings, params.itters, trainset, devset, testset)
        trainer.start_training(params.output_base, batch_size=params.batch_size)

    elif params.train == "compound":
        print("Starting training for " + params.train)
        print("==PARAMETERS==")
        print("TRAIN FILE: " + params.train_file)
        print("DEV FILE: " + params.dev_file)
        if params.test_file is not None:
            print("TEST FILE: " + params.test_file)
        print("EMBEDDINGS FILE: " + params.embeddings)
        print("STOPPING CONDITION: " + str(params.itters))
        print("OUTPUT BASE: " + params.output_base)
        print("AUX SOFTMAX WEIGHT: " + str(params.aux_softmax_weight))
        print("CONFIG FILE: " + str(params.config))
        print("==============\n")

        trainset = Dataset(params.train_file)
        devset = Dataset(params.dev_file)
        if params.test_file:
            testset = Dataset(params.test_file)
        else:
            testset = None
        config = CompoundWordConfig(params.config)
        encodings = Encodings()
        encodings.compute(trainset, devset, 'label')
        # update wordlist if testset was provided
        if params.test_file:
            encodings.update_wordlist(testset)

        embeddings = None
        expander = CompoundWordExpander(config, encodings, embeddings)
        trainer = CompoundWordTrainer(expander, encodings, params.itters, trainset, devset, testset)
        trainer.start_training(params.output_base, batch_size=params.batch_size)

    elif params.train == "tokenizer":
        print("Starting training for " + params.train)
        print("==PARAMETERS==")
        print("TRAIN FILE: " + params.train_file)
        print("RAW TRAIN FILE: " + (params.raw_train_file if params.raw_train_file is not None else "n/a"))
        print("DEV FILE: " + params.dev_file)
        print("RAW DEV FILE: " + (params.raw_dev_file if params.raw_dev_file is not None else "n/a"))
        print("TEST FILE: " + (params.test_file if params.test_file is not None else "n/a"))
        print("RAW TEST FILE: " + (params.raw_test_file if params.raw_test_file is not None else "n/a"))
        print("EMBEDDINGS FILE: " + params.embeddings)
        print("STOPPING CONDITION: " + str(params.itters))
        print("OUTPUT BASE: " + params.output_base)
        print("CONFIG FILE: " + str(params.config))
        print("==============\n")
        trainset = Dataset(params.train_file)
        devset = Dataset(params.dev_file)
        if params.test_file:
            testset = Dataset(params.test_file)
        else:
            testset = None
        from generic_networks.tokenizers import TieredTokenizer
        config = TieredTokenizerConfig(params.config)
        config.raw_test_file = params.raw_test_file
        config.base = params.output_base
        config.patience = params.itters
        if not config._valid:
            return

        encodings = Encodings()
        encodings.compute(trainset, devset, 'label')
        # update wordlist if testset was provided
        if params.test_file:
            encodings.update_wordlist(testset)
        embeddings = WordEmbeddings()
        embeddings.read_from_file(params.embeddings,
                                  None)  # setting wordlist to None triggers Word Embeddings to act as cache-only and load offsets for all words
        tokenizer = TieredTokenizer(config, encodings, embeddings)
        trainer = TokenizerTrainer(tokenizer, encodings, params.itters, trainset, devset, testset,
                                   raw_train_file=params.raw_train_file, raw_dev_file=params.raw_dev_file,
                                   raw_test_file=params.raw_test_file, gold_train_file=params.train_file,
                                   gold_dev_file=params.dev_file, gold_test_file=params.test_file)
        trainer.start_training(params.output_base, batch_size=params.batch_size)
    elif params.train == "ner":
        print ("Starting training for " + params.train)
        print ("==PARAMETERS==")
        print ("TRAIN FILE: " + params.train_file)
        print ("DEV FILE: " + params.dev_file)
        if params.test_file is not None:
            print ("TEST FILE: " + params.test_file)
        print ("EMBEDDINGS FILE: " + params.embeddings)
        print ("STOPPING CONDITION: " + str(params.itters))
        print ("OUTPUT BASE: " + params.output_base)
        print ("AUX SOFTMAX WEIGHT: " + str(params.aux_softmax_weight))
        print ("CONFIG FILE: " + str(params.config))
        print ("==============\n")

        trainset = CUPTDataset(params.train_file)
        devset = CUPTDataset(params.dev_file)
        if params.test_file:
            testset = CUPTDataset(params.test_file)
        else:
            testset = None
        config = GDBConfig(params.config)
        if not config._valid:
            return
        # PARAM INJECTION
        if params.params != None:
            parts = params.params.split(":")
            for param in parts:
                variable = param.split("=")[0]
                value = param[len(variable) + 1:]
                print("External param injection: " + variable + "=" + value)
                exec ("config.__dict__[\"" + variable + "\"] = " + value)
                # END INJECTION
        encodings = Encodings()
        encodings.compute(trainset, devset, 'label', CUPT_format=True)
        # update wordlist if testset was provided
        if params.test_file:
            encodings.update_wordlist(testset)
        embeddings = WordEmbeddings()
        embeddings.read_from_file(params.embeddings, None)
        gdbner = GDBNer(config, encodings, embeddings)
        trainer = NERTrainer(gdbner, encodings, params.itters, trainset, devset, testset)
        trainer.start_training(params.output_base, params.batch_size)


def parse_run(params):
    sys.stdout.write("\nINPUT FILE: " + params.input_file)
    sys.stdout.write("\nOUTPUT FILE: " + params.output_file)
    sys.stdout.write("\nMODELS FILE: " + params.models + "\n")
    sys.stdout.flush()

    components = params.run.split(",")
    tokenize = True if "tokenizer" in components else False
    compound = True if "compound" in components else False
    lemmatize = True if "lemmatizer" in components else False
    tag = True if "tagger" in components else False
    parse = True if "parser" in components else False

    # common elements load
    sys.stdout.write("\nLoading embeddings : " + params.embeddings + " ...\n")
    embeddings = WordEmbeddings()
    embeddings.read_from_file(params.embeddings, None)

    encodings = None
    if tokenize == True:
        if not os.path.isfile(os.path.join(params.models, "tokenizer-tok.bestAcc")):
            sys.stdout.write(
                "\n\tTokenizer model not found! (" + os.path.join(params.models, "tokenizer-tok.bestAcc") + ")")
            sys.stdout.flush()
            sys.exit(1)
        sys.stdout.write("\n\tTokenization enabled.\n")
        tokenizer_encodings = Encodings(verbose=False)
        tokenizer_encodings.load(os.path.join(params.models, "tokenizer.encodings"))
    if compound == True:
        if not os.path.isfile(os.path.join(params.models, "compound.bestAcc")):
            sys.stdout.write("\n\tCompound word expander model not found!")
            sys.stdout.flush()
            sys.exit(1)
        sys.stdout.write("\n\tCompound word expander enabled.\n")
        if encodings == None:
            encodings = Encodings(verbose=False)
            encodings.load(os.path.join(params.models, "lemmatizer.encodings"))
    if lemmatize == True:
        if not os.path.isfile(os.path.join(params.models, "lemmatizer.bestACC")):
            sys.stdout.write("\n\tLemmatization model not found!")
            sys.stdout.flush()
            sys.exit(1)
        sys.stdout.write("\n\tLemmatization enabled.\n")
        if encodings == None:
            encodings = Encodings(verbose=False)
            encodings.load(os.path.join(params.models, "lemmatizer.encodings"))
    if tag == True:
        if not os.path.isfile(os.path.join(params.models, "tagger.bestOVERALL")):
            sys.stdout.write("\n\tTagger model not found!")
            sys.stdout.flush()
            sys.exit(1)
        sys.stdout.write("\n\tTagger enabled.\n")
        if encodings == None:
            encodings = Encodings(verbose=False)
            encodings.load(os.path.join(params.models, "tagger.encodings"))
    if parse == True:
        if not os.path.isfile(os.path.join(params.models, "parser.bestUAS")):
            sys.stdout.write("\n\tParser model not found!")
            sys.stdout.flush()
            sys.exit(1)
        sys.stdout.write("\n\tParser enabled.\n")
        if encodings == None:
            encodings = Encodings(verbose=False)
            encodings.load(os.path.join(params.models, "parser.encodings"))

    sequences = None
    if tokenize:
        sys.stdout.write("\nTokenizing " + params.input_file + " ... \n\t")
        sys.stdout.flush()

        from io_utils.config import TieredTokenizerConfig
        from generic_networks.tokenizers import TieredTokenizer
        config = TieredTokenizerConfig(os.path.join(params.models, "tokenizer.conf"))
        tokenizer_object = TieredTokenizer(config, tokenizer_encodings, embeddings, runtime=True)
        tokenizer_object.load(os.path.join(params.models, "tokenizer"))

        with fopen(params.input_file, 'r') as file:
            lines = file.readlines()
        # analyze use of spaces in first part of the file
        test = "";
        useSpaces = " "
        cnt = 0
        while True:
            test = test + lines[cnt]
            # print(lines[cnt])
            if cnt >= len(lines) or cnt > 5:
                break
            cnt += 1
        if float(test.count(' ')) / float(len(test)) < 0.02:
            useSpaces = ""
        # print (str(float(test.count(' '))/float(len(test))))
        i = -1
        input_string = ""
        sequences = []
        while i < len(lines) - 1:
            i += 1
            input_string = input_string + lines[i].replace("\r", "").replace("\n", "").strip() + useSpaces
            if lines[i].strip() == "" or i == len(lines) - 1:  # end of block
                if input_string.strip() != "":
                    sequences += tokenizer_object.tokenize(input_string)
                input_string = ""

        del tokenizer_object  # free memory
    else:
        ds = Dataset(params.input_file)
        sequences = ds.sequences
    sys.stdout.write(" done\n")
    sys.stdout.flush()

    if compound:
        sys.stdout.write("\nCompound word expanding " + params.input_file + " ... \n\t")
        sys.stdout.flush()
        from generic_networks.token_expanders import CompoundWordExpander
        from io_utils.config import CompoundWordConfig
        config = CompoundWordConfig(os.path.join(params.models, "compound.conf"))
        compoundwordexpander_object = CompoundWordExpander(config, encodings, embeddings, runtime=True)
        compoundwordexpander_object.load(os.path.join(params.models, "compound.bestAcc"))
        sequences = compoundwordexpander_object.expand_sequences(sequences)
        del compoundwordexpander_object  # free memory
        sys.stdout.write(" done\n")
        sys.stdout.flush()

    if parse == True:
        sys.stdout.write("\nParsing " + params.input_file + " ... \n\t")
        sys.stdout.flush()
        from io_utils.config import ParserConfig
        from generic_networks.parsers import BDRNNParser
        config = ParserConfig(os.path.join(params.models, "parser.conf"))
        parser_object = BDRNNParser(config, encodings, embeddings, runtime=True)
        parser_object.load(os.path.join(params.models, "parser.bestUAS"))
        sequences = parser_object.parse_sequences(sequences)
        del parser_object  # free memory
        sys.stdout.write(" done\n")
        sys.stdout.flush()

    if tag == True:
        sys.stdout.write("\nTagging " + params.input_file + " ... \n\t")
        sys.stdout.flush()
        from io_utils.config import TaggerConfig
        from generic_networks.taggers import BDRNNTagger
        config = TaggerConfig(os.path.join(params.models, "tagger.conf"))
        tagger_object_UPOS = BDRNNTagger(config, encodings, embeddings, runtime=True)
        tagger_object_UPOS.load(os.path.join(params.models, "tagger.bestUPOS"))
        tagger_object_XPOS = BDRNNTagger(config, encodings, embeddings, runtime=True)
        tagger_object_XPOS.load(os.path.join(params.models, "tagger.bestXPOS"))
        tagger_object_ATTRS = BDRNNTagger(config, encodings, embeddings, runtime=True)
        tagger_object_ATTRS.load(os.path.join(params.models, "tagger.bestATTRS"))

        new_sequences = []
        for sequence in sequences:
            new_sequence = copy.deepcopy(sequence)
            predicted_tags_UPOS = tagger_object_UPOS.tag(new_sequence)
            predicted_tags_XPOS = tagger_object_XPOS.tag(new_sequence)
            predicted_tags_ATTRS = tagger_object_ATTRS.tag(new_sequence)
            for entryIndex in range(len(sequence)):
                new_sequence[entryIndex].upos = predicted_tags_UPOS[entryIndex][0]
                new_sequence[entryIndex].xpos = predicted_tags_XPOS[entryIndex][1]
                new_sequence[entryIndex].attrs = predicted_tags_ATTRS[entryIndex][2]
            new_sequences.append(new_sequence)
        sequences = copy.deepcopy(new_sequences)
        del tagger_object_UPOS  # free memory
        del tagger_object_XPOS  # free memory
        del tagger_object_ATTRS  # free memory
        sys.stdout.write(" done\n")
        sys.stdout.flush()

    if lemmatize:
        sys.stdout.write("\nLemmatizing " + params.input_file + " ... \n\t")
        sys.stdout.flush()
        from generic_networks.lemmatizers import FSTLemmatizer
        from io_utils.config import LemmatizerConfig
        config = LemmatizerConfig(os.path.join(params.models, "lemmatizer.conf"))
        lemmatizer_object = FSTLemmatizer(config, encodings, embeddings, runtime=True)
        lemmatizer_object.load(os.path.join(params.models, "lemmatizer.bestACC"))
        sequences = lemmatizer_object.lemmatize_sequences(sequences)
        del lemmatizer_object  # free memory
        sys.stdout.write(" done\n")
        sys.stdout.flush()

    output_dataset = Dataset()
    output_dataset.sequences = sequences
    output_dataset.write(params.output_file)


if params.train:
    valid = True
    if params.train != 'mt':
        if not params.train_file:
            print ("--train-file is mandatory")
            valid = False
        if not params.dev_file:
            print ("--dev-file is mandatory")
            valid = False
        if not params.embeddings:
            print ("--embeddings is mandatory")
            valid = False
        if not params.output_base:
            print ("--store is mandatory")
            valid = False
    if valid:
        parse_train(params)

if params.server:
    from server.webserver import EmbeddedWebserver

    WordEmbeddings
    we = WordEmbeddings()
    we.read_from_file(params.embeddings, None, False)
    ews = EmbeddedWebserver(we, port=params.port, lemma=params.model_lemmatization,
                            tokenization=params.model_tokenization, tagging=params.model_tagging,
                            parsing=params.model_parsing)

if params.test:
    valid = True
    if valid:
        parse_test(params)

if params.run:
    valid = True
    if "all" in params.run:
        params.run = "tokenizer,compound,parser,tagger,lemmatizer"
    components = params.run.split(",")
    if len(components) == 0:
        print ("--run needs a list of components")
        valid = False
    if not params.embeddings:
        print ("--embeddings is mandatory")
        valid = False
    if not params.models:
        print ("--models is mandatory")
        valid = False
    if not params.input_file:
        print ("--input-file is mandatory")
        valid = False
    if not params.output_file:
        print ("--output-file is mandatory")
        valid = False
    if "lemmatizer" in params.run and not "tagger" in params.run:
        print (
            "Warning: --run needs to include a tagger to be able to lemmatize words. If not provided ensure the input file is in conllu format and includes POS tags.")
        valid = True
    if valid:
        parse_run(params)
