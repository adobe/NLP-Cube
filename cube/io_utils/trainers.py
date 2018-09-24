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
from misc.misc import fopen

sys.path.insert(0, '../')
from random import shuffle
import time
import random
from cube.misc.conll18_ud_eval_wrapper import conll_eval


# import nltk


class MTTrainer:
    def __init__(self, translator, src_enc, dst_enc, src_we, dst_we, patience, trainset, devset, testset=None):
        self.translator = translator
        self.src_enc = src_enc
        self.dst_enc = dst_enc
        self.src_we = src_we
        self.dst_we = dst_we
        self.patience = patience
        self.trainset = trainset
        self.devset = devset
        self.testset = testset

    def start_training(self, output_base, batch_size=100):
        itt_no_improve = self.patience
        selected_test_BLEU = 0
        selected_dev_BLEU = 0
        path = output_base + ".src.encodings"
        sys.stdout.write("Storing source encodings in " + path + "\n")
        self.src_enc.save(path)
        path = output_base + ".conf"
        sys.stdout.write("Storing config in " + path + "\n")
        self.translator.config.save(path)
        path = output_base + ".dst.encodings"
        sys.stdout.write("Storing destination encodings in " + path + "\n")
        self.dst_enc.save(path)
        epoch = 0
        # sys.stdout.write("\tevaluating on devset...")
        # sys.stdout.flush()
        # bleu_dev = self.eval(self.devset)
        # sys.stdout.write(" BLEU=" + str(bleu_dev) + "\n")

        while itt_no_improve > 0:
            itt_no_improve -= 1
            epoch += 1
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.write("\tshuffling training data... ")
            sys.stdout.flush()
            shuffle(self.trainset.sequences)
            sys.stdout.write("done\n")
            sys.stdout.flush()
            sys.stdout.write("\ttraining...")
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()
            cbs = 0
            self.translator.start_batch()
            last_proc = 0
            for iSeq in range(len(self.trainset.sequences)):
                seq = self.trainset.sequences[iSeq]
                proc = int((iSeq + 1) * 100 / len(self.trainset.sequences))
                if proc % 5 == 0 and proc != last_proc:
                    last_proc = proc
                    sys.stdout.write(" " + str(proc))
                    sys.stdout.flush()

                self.translator.learn(seq.src, seq.dst)
                cbs += len(seq.src) + len(seq.dst)
                if cbs >= batch_size:
                    total_loss += self.translator.end_batch()
                    self.translator.start_batch()

            total_loss += self.translator.end_batch()
            stop_time = time.time()
            sys.stdout.write(" avg_loss=" + str(total_loss / len(self.trainset.sequences)) + " execution_time=" + str(
                stop_time - start_time) + "\n")

            sys.stdout.write("\tevaluating on devset...")
            sys.stdout.flush()
            bleu_dev = self.eval(self.devset, filename=output_base + ".dev.out")
            sys.stdout.write(" BLEU=" + str(bleu_dev) + "\n")

            if self.testset is not None:
                sys.stdout.write("\tevaluating on testset...")
                sys.stdout.flush()
                bleu_test = self.eval(self.testset, filename=output_base + ".test.out")
                sys.stdout.write(" BLEU=" + str(bleu_test) + "\n")
            else:
                bleu_test = 0
            if bleu_dev > selected_dev_BLEU:
                selected_dev_BLEU = bleu_dev
                selected_test_BLEU = bleu_test
                itt_no_improve = self.patience
                self.translator.save(output_base + ".bestBLEU")
            self.translator.save(output_base + ".last")

        sys.stdout.write("Training is done with devset BLEU=" + str(selected_dev_BLEU) + " and testset BLEU=" + str(
            selected_test_BLEU) + " (for the selected devset)\n")

    def eval(self, dataset, filename=None):
        total_bleu = 0.0
        last_proc = 0
        iSeq = 0
        if filename is not None:
            f = fopen(filename, "w", encoding="utf-8")

        for seq in dataset.sequences:
            proc = int((iSeq + 1) * 100 / len(dataset.sequences))
            if proc % 5 == 0 and proc != last_proc:
                last_proc = proc
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()
            iSeq += 1

            hyp = self.translator.translate(seq.src)
            ref = [entry.word for entry in seq.dst]
            hyp = list(hyp)
            ref = list(ref)
            # print "hyp=",hyp
            # print "ref=",ref
            # print "\n\n\n\n"
            # sys.stdout.flush()
            if filename is not None:
                for entry in seq.src:
                    f.write(entry.word + " ")
                f.write("\n")
                for entry in seq.dst:
                    f.write(entry.word + " ")
                f.write("\n")
                for word in hyp:
                    f.write(word.encode('utf-8') + " ")
                f.write("\n\n")

            if len(ref) >= 4 and len(hyp) >= 4:
                score = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
                total_bleu += score
        if filename is not None:
            f.close()
        return total_bleu / len(dataset.sequences)


class LemmatizerTrainer:
    def __init__(self, lemmatizer, encodings, patience, trainset, devset, testset=None):
        self.tagger = lemmatizer
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.patience = patience
        self.encodings = encodings

    def start_training(self, output_base, batch_size=1):
        epoch = 0
        itt_no_improve = self.patience
        selected_test_acc = 0
        selected_dev_acc = 0
        path = output_base + ".encodings"
        sys.stdout.write("Storing encodings in " + path + "\n")
        self.encodings.save(path)
        path = output_base + ".conf"
        sys.stdout.write("Storing config in " + path + "\n")
        self.tagger.config.save(path)
        sys.stdout.write("\tevaluating on devset...")
        sys.stdout.flush()
        dev_acc = 0  # self.eval(self.devset)
        sys.stdout.write(" accuracy=" + str(dev_acc) + "\n")
        if self.testset is not None:
            sys.stdout.write("\tevaluating on testset...")
            sys.stdout.flush()
            test_acc = 0  # self.eval(self.testset)
            sys.stdout.write(" accuracy=" + str(test_acc) + "\n")
        best_dev_acc = dev_acc

        while itt_no_improve > 0:
            itt_no_improve -= 1
            epoch += 1
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.flush()
            sys.stdout.write("\tshuffling training data... ")
            sys.stdout.flush()
            shuffle(self.trainset.sequences)
            sys.stdout.write("done\n")
            sys.stdout.flush()
            last_proc = 0
            sys.stdout.write("\ttraining...")
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()
            current_batch_size = 0
            self.tagger.start_batch()
            for iSeq in range(len(self.trainset.sequences)):
                seq = self.trainset.sequences[iSeq]
                proc = int((iSeq + 1) * 100 / len(self.trainset.sequences))
                if proc % 5 == 0 and proc != last_proc:
                    last_proc = proc
                    sys.stdout.write(" " + str(proc))
                    sys.stdout.flush()

                self.tagger.learn(seq)
                current_batch_size += len(seq)
                if current_batch_size >= batch_size:
                    total_loss += self.tagger.end_batch()
                    self.tagger.start_batch()
                    current_batch_size = 0
            total_loss += self.tagger.end_batch()
            stop_time = time.time()
            sys.stdout.write(" avg_loss=" + str(total_loss / len(self.trainset.sequences)) + " execution_time=" + str(
                stop_time - start_time) + "\n")

            sys.stdout.write("\tevaluating on devset...")
            sys.stdout.flush()
            dev_acc = self.eval(self.devset)
            sys.stdout.write(" accuracy=" + str(dev_acc) + "\n")
            if self.testset is not None:
                sys.stdout.write("\tevaluating on testset...")
                sys.stdout.flush()
                test_acc = self.eval(self.testset)
                sys.stdout.write(" accuracy=" + str(test_acc) + ")\n")

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                selected_dev_acc = dev_acc
                if self.testset is not None:
                    selected_test_acc = test_acc
                path = output_base + ".bestAcc"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.tagger.save(path)
                itt_no_improve = self.patience

            path = output_base + ".last"
            sys.stdout.write("\tStoring " + path + "\n")
            sys.stdout.flush()
            self.tagger.save(path)

        sys.stdout.write("Training is done with devset_accuracy=" + str(selected_dev_acc) + "\n")
        if self.testset is not None:
            sys.stdout.write(" and testset_accuracy=" + str(
                selected_test_acc) + " (for the selected epoch, based on best devset accuracy)")
        sys.stdout.write("\n")

    def eval(self, dataset):
        last_proc = 0
        correct = 0
        total = 0

        for iSeq in range(len(dataset.sequences)):
            seq = dataset.sequences[iSeq]

            proc = int((iSeq + 1) * 100 / len(dataset.sequences))
            if proc % 5 == 0 and proc != last_proc:
                last_proc = proc
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()

            pred_lemmas = self.tagger.tag(seq)

            for entry, pred_lemma in zip(seq, pred_lemmas):
                if entry.upos != 'NUM' and entry.upos != 'PROPN':
                    total += 1
                    # from pdb import set_trace
                    # set_trace()
                    if sys.version_info[0] == 2:
                        if unicode(entry.lemma, 'utf-8') == pred_lemma:
                            correct += 1
                    else:
                        if entry.lemma == pred_lemma:
                            correct += 1
                else:
                    correct += 1
                    total += 1

        return float(correct) / total


class CompoundWordTrainer:
    def __init__(self, cw, encodings, patience, trainset, devset, testset=None):
        self.tagger = cw
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.patience = patience
        self.encodings = encodings

    def start_training(self, output_base, batch_size=1):
        epoch = 0
        itt_no_improve = self.patience
        selected_test_acc = 0
        selected_dev_acc = 0
        path = output_base + ".encodings"
        sys.stdout.write("Storing encodings in " + path + "\n")
        self.encodings.save(path)
        path = output_base + ".conf"
        sys.stdout.write("Storing config in " + path + "\n")
        self.tagger.config.save(path)
        sys.stdout.write("\tevaluating on devset...")
        sys.stdout.flush()
        dev_fscore = 0  # self.eval(self.devset)
        dev_acc = 0
        sys.stdout.write(" accuracy=" + str(dev_acc) + "\n")
        if self.testset is not None:
            sys.stdout.write("\tevaluating on testset...")
            sys.stdout.flush()
            test_fscore = 0
            test_acc = 0  # self.eval(self.testset)
            sys.stdout.write(" accuracy=" + str(test_acc) + "\n")
        best_dev_acc = dev_acc

        while itt_no_improve > 0:
            itt_no_improve -= 1
            epoch += 1
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.flush()
            sys.stdout.write("\tshuffling training data... ")
            sys.stdout.flush()
            shuffle(self.trainset.sequences)
            sys.stdout.write("done\n")
            sys.stdout.flush()
            last_proc = 0
            sys.stdout.write("\ttraining...")
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()
            current_batch_size = 0
            self.tagger.start_batch()
            for iSeq in range(len(self.trainset.sequences)):
                seq = self.trainset.sequences[iSeq]
                proc = int((iSeq + 1) * 100 / len(self.trainset.sequences))
                if proc % 5 == 0 and proc != last_proc:
                    last_proc = proc
                    sys.stdout.write(" " + str(proc))
                    sys.stdout.flush()

                self.tagger.learn(seq)
                current_batch_size += len(seq)
                if current_batch_size >= batch_size:
                    total_loss += self.tagger.end_batch()
                    self.tagger.start_batch()
                    current_batch_size = 0
            total_loss += self.tagger.end_batch()
            stop_time = time.time()
            sys.stdout.write(" avg_loss=" + str(total_loss / len(self.trainset.sequences)) + " execution_time=" + str(
                stop_time - start_time) + "\n")

            sys.stdout.write("\tevaluating on devset...")
            sys.stdout.flush()
            dev_fscore, dev_acc = self.eval(self.devset)
            sys.stdout.write(" fscore=" + str(dev_fscore) + " accuracy=" + str(dev_acc) + "\n")
            if self.testset is not None:
                sys.stdout.write("\tevaluating on testset...")
                sys.stdout.flush()
                test_fscore, test_acc = self.eval(self.testset)
                sys.stdout.write(" fscore=" + str(test_fscore) + " accuracy=" + str(test_acc) + ")\n")

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                selected_dev_acc = dev_acc
                selected_dev_fscore = dev_fscore
                if self.testset is not None:
                    selected_test_acc = test_acc
                    selected_test_fscore = test_fscore
                path = output_base + ".bestAcc"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.tagger.save(path)
                itt_no_improve = self.patience

            path = output_base + ".last"
            sys.stdout.write("\tStoring " + path + "\n")
            sys.stdout.flush()
            self.tagger.save(path)

        sys.stdout.write(
            "Training is done with devset fscore=" + str(selected_dev_fscore) + " acc=" + str(selected_dev_acc) + "\n")
        if self.testset is not None:
            sys.stdout.write(" and testset fscore=" + str(
                selected_test_fscore) + " acc=" + str(
                selected_test_acc) + " (for the selected epoch, based on best devset accuracy)")
        sys.stdout.write("\n")

    def eval(self, dataset):
        detection_correct = 0
        detection_incorrect = 0
        detection_total = 0
        detection_real = 0

        tokens_correct = 0
        tokens_total = 0

        last_proc = 0

        for iSeq in range(len(dataset.sequences)):
            seq = dataset.sequences[iSeq]

            proc = int((iSeq + 1) * 100 / len(dataset.sequences))
            if proc % 5 == 0 and proc != last_proc:
                last_proc = proc
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()

            i_entry = 0
            while i_entry < len(seq):
                entry = seq[i_entry]
                if entry.is_compound_entry:
                    detection_real += 1

                    compound, tokens = self.tagger.tag_token(entry.word)
                    if compound:
                        detection_correct += 1
                        detection_total += 1
                    interval = entry.index.split("-")
                    interval = int(interval[1]) - int(interval[0]) + 1
                    real_tokens = []
                    for _ in range(interval):
                        i_entry += 1
                        real_tokens.append(seq[i_entry].word)
                    i_entry += 1
                    tokens_total += len(real_tokens)
                    for pt, rt in zip(tokens, real_tokens):
                        if sys.version_info[0] == 2:
                            if pt.encode('utf-8') == rt:
                                tokens_correct += 1
                        else:
                            if pt == rt:
                                tokens_correct += 1

                else:
                    compound, _ = self.tagger.tag_token(entry.word)
                    if compound:
                        detection_incorrect += 1
                        detection_total += 1
                i_entry += 1
        if detection_total == 0:
            detection_total += 1
        if detection_real == 0:
            detection_real += 1
        p = float(detection_correct) / detection_total
        r = float(detection_correct) / detection_real
        if p == 0 or r == 0:
            f = 0
        else:
            f = 2 * p * r / (p + r)

        acc = float(tokens_correct) / tokens_total
        return f, acc


class TaggerTrainer:
    def __init__(self, tagger, encodings, patience, trainset, devset, testset=None):
        self.tagger = tagger
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.patience = patience
        self.encodings = encodings

    def start_training(self, output_base, batch_size=0):
        epoch = 0
        itt_no_improve = self.patience
        selected_test_upos, selected_test_xpos, selected_test_attrs = 0, 0, 0
        selected_dev_upos, selected_dev_xpos, selected_dev_attrs = 0, 0, 0
        path = output_base + ".encodings"
        sys.stdout.write("Storing encodings in " + path + "\n")
        self.encodings.save(path)
        path = output_base + ".conf"
        sys.stdout.write("Storing config in " + path + "\n")
        self.tagger.config.save(path)
        sys.stdout.write("\tevaluating on devset...")
        sys.stdout.flush()
        dev_upos, dev_xpos, dev_attrs = self.eval(self.devset)
        sys.stdout.write(" accuracy=( UPOS=" + str(dev_upos) + " , XPOS=" + str(dev_xpos) + " , ATTRS=" + str(
            dev_attrs) + " )\n")
        if self.testset is not None:
            sys.stdout.write("\tevaluating on testset...")
            sys.stdout.flush()
            test_acc = self.eval(self.testset)
            sys.stdout.write(" accuracy=" + str(test_acc) + "\n")
        best_dev_upos = dev_upos
        best_dev_xpos = dev_upos
        best_dev_attrs = dev_upos
        best_dev_overall = (dev_upos + dev_xpos + dev_attrs) / 3

        while itt_no_improve > 0:
            itt_no_improve -= 1
            epoch += 1
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.flush()
            sys.stdout.write("\tshuffling training data... ")
            sys.stdout.flush()
            shuffle(self.trainset.sequences)
            sys.stdout.write("done\n")
            sys.stdout.flush()
            last_proc = 0
            sys.stdout.write("\ttraining...")
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()
            current_batch_size = 0
            self.tagger.start_batch()
            for iSeq in range(len(self.trainset.sequences)):
                seq = self.trainset.sequences[iSeq]
                proc = int((iSeq + 1) * 100 / len(self.trainset.sequences))
                if proc % 5 == 0 and proc != last_proc:
                    last_proc = proc
                    sys.stdout.write(" " + str(proc))
                    sys.stdout.flush()

                self.tagger.learn(seq)
                current_batch_size += len(seq)
                if current_batch_size > batch_size:
                    total_loss += self.tagger.end_batch()
                    self.tagger.start_batch()
                    current_batch_size = 0

            if current_batch_size != 0:
                total_loss += self.tagger.end_batch()
                self.tagger.start_batch()

            stop_time = time.time()
            sys.stdout.write(" avg_loss=" + str(total_loss / len(self.trainset.sequences)) + " execution_time=" + str(
                stop_time - start_time) + "\n")

            sys.stdout.write("\tevaluating on trainset...")
            sys.stdout.flush()
            train_upos, train_xpos, train_attrs = self.eval(self.trainset)
            sys.stdout.write(" accuracy=( UPOS=" + str(train_upos) + " , XPOS=" + str(train_xpos) + " , ATTRS=" + str(
                train_attrs) + " )\n")

            sys.stdout.write("\tevaluating on devset...")
            sys.stdout.flush()
            dev_upos, dev_xpos, dev_attrs = self.eval(self.devset)
            sys.stdout.write(" accuracy=( UPOS=" + str(dev_upos) + " , XPOS=" + str(dev_xpos) + " , ATTRS=" + str(
                dev_attrs) + " )\n")
            if self.testset is not None:
                sys.stdout.write("\tevaluating on testset...")
                sys.stdout.flush()
                test_upos, test_xpos, test_attrs = self.eval(self.testset)
                sys.stdout.write(" accuracy=( UPOS=" + str(test_upos) + " , XPOS=" + str(test_xpos) + " , ATTRS=" + str(
                    test_attrs) + " )\n")

            if (dev_upos + dev_xpos + dev_attrs) / 3 > best_dev_overall:
                best_dev_overall = (dev_upos + dev_xpos + dev_attrs) / 3
                selected_dev_upos = dev_upos
                selected_dev_xpos = dev_xpos
                selected_dev_attrs = dev_attrs
                if self.testset is not None:
                    selected_test_upos = test_upos
                    selected_test_xpos = test_xpos
                    selected_test_attrs = test_attrs
                path = output_base + ".bestOVERALL"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.tagger.save(path)
                itt_no_improve = self.patience

            if dev_upos > best_dev_upos:
                best_dev_upos = dev_upos
                path = output_base + ".bestUPOS"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.tagger.save(path)
                itt_no_improve = self.patience

            if dev_xpos > best_dev_xpos:
                best_dev_xpos = dev_xpos
                path = output_base + ".bestXPOS"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.tagger.save(path)
                itt_no_improve = self.patience

            if dev_attrs > best_dev_attrs:
                best_dev_attrs = dev_attrs
                path = output_base + ".bestATTRS"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.tagger.save(path)
                itt_no_improve = self.patience

            path = output_base + ".last"
            sys.stdout.write("\tStoring " + path + "\n")
            sys.stdout.flush()
            self.tagger.save(path)

        sys.stdout.write("Training is done with devset accuracy=( UPOS=" + str(selected_dev_upos) + " , XPOS=" + str(
            selected_dev_xpos) + " , ATTRS=" + str(selected_dev_attrs) + " )\n")
        if self.testset is not None:
            sys.stdout.write(" and testset_accuracy=( UPOS=" + str(selected_test_upos) + " , XPOS=" + str(
                selected_test_xpos) + " , ATTRS=" + str(
                selected_test_attrs) + " ) (for the selected epoch, based on best devset accuracy)")
        sys.stdout.write("\n")

    def eval(self, dataset):
        last_proc = 0
        correct_upos = 0
        correct_xpos = 0
        correct_attrs = 0
        total = 0

        for iSeq in range(len(dataset.sequences)):
            seq = dataset.sequences[iSeq]

            proc = int((iSeq + 1) * 100 / len(dataset.sequences))
            if proc % 5 == 0 and proc != last_proc:
                last_proc = proc
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()

            pred_tags = self.tagger.tag(seq)

            for entry, pred_tag in zip(seq, pred_tags):
                total += 1

                if entry.upos == pred_tag[0]:
                    correct_upos += 1
                if entry.xpos == pred_tag[1]:
                    correct_xpos += 1
                if entry.attrs == pred_tag[2]:
                    correct_attrs += 1

        return float(correct_upos) / total, float(correct_xpos) / total, float(correct_attrs) / total


class ParserTrainer:
    def __init__(self, parser, encodings, patience, trainset, devset, testset=None):
        self.parser = parser
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.patience = patience
        self.encodings = encodings

    def start_training(self, output_base, batch_size=100):

        epoch = 0
        itt_no_improve = self.patience

        path = output_base + ".encodings"
        sys.stdout.write("Storing encodings in " + path + "\n")
        self.encodings.save(path)
        path = output_base + ".conf"
        sys.stdout.write("Storing config in " + path + "\n")
        self.parser.config.save(path)
        sys.stdout.write("\tevaluating on devset...")
        sys.stdout.flush()
        # dev_uas, dev_las, dev_upos, dev_xpos, dev_attrs, dev_lemma = self.eval(self.devset)
        # sys.stdout.write(" UAS=" + str(dev_uas) + " LAS=" + str(dev_las) + " UPOS=" + str(dev_upos) + " XPOS=" + str(
        #     dev_xpos) + " ATTRS=" + str(dev_attrs) + " LEMMA=" + str(dev_lemma) + "\n")
        # if self.testset is not None:
        #     sys.stdout.write("\tevaluating on testset...")
        #     sys.stdout.flush()
        #     test_uas, test_las, test_upos, test_xpos, test_attrs, test_lemma = self.eval(self.testset)
        #     sys.stdout.write(
        #         " UAS=" + str(test_uas) + " LAS=" + str(test_las) + " UPOS=" + str(test_upos) + " XPOS=" + str(
        #             test_xpos) + " ATTRS=" + str(test_attrs) + " LEMMA=" + str(test_lemma) + "\n")

        # best_dev_uas = dev_uas
        # best_dev_las = dev_las
        best_dev_uas = 0
        best_dev_las = 0
        test_uas_uas = 0
        test_uas_las = 0
        test_las_uas = 0
        test_las_las = 0
        dev_uas_uas = 0
        dev_uas_las = 0
        dev_las_uas = 0
        dev_las_las = 0
        current_batch_size = 0
        self.parser.start_batch()

        while itt_no_improve > 0:

            itt_no_improve -= 1
            epoch += 1
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.flush()
            sys.stdout.write("\tshuffling training data... ")
            sys.stdout.flush()
            shuffle(self.trainset.sequences)
            sys.stdout.write("done\n")
            sys.stdout.flush()
            last_proc = 0
            sys.stdout.write("\ttraining...")
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()

            for iSeq in range(len(self.trainset.sequences)):
                seq = self.trainset.sequences[iSeq]
                proc = int((iSeq + 1) * 100 / len(self.trainset.sequences))
                if proc % 5 == 0 and proc != last_proc:
                    last_proc = proc
                    sys.stdout.write(" " + str(proc))
                    sys.stdout.flush()

                self.parser.learn(seq)
                current_batch_size += len(seq)
                if current_batch_size >= batch_size:
                    total_loss += self.parser.end_batch()
                    current_batch_size = 0
                    self.parser.start_batch()
            total_loss += self.parser.end_batch()
            current_batch_size = 0
            stop_time = time.time()
            sys.stdout.write(" avg_loss=" + str(total_loss / len(self.trainset.sequences)) + " execution_time=" + str(
                stop_time - start_time) + "\n")
            self.parser.start_batch()

            # sys.stdout.write("\tevaluating on trainset...")
            # sys.stdout.flush()
            # train_uas, train_las = self.eval(self.trainset)
            # sys.stdout.write(" UAS=" + str(train_uas) + " LAS=" + str(train_las) + "\n")

            sys.stdout.write("\tevaluating on devset...")
            sys.stdout.flush()
            dev_uas, dev_las, dev_upos, dev_xpos, dev_attrs = self.eval(self.devset)
            sys.stdout.write(
                " UAS=" + str(dev_uas) + " LAS=" + str(dev_las) + " UPOS=" + str(dev_upos) + " XPOS=" + str(
                    dev_xpos) + " ATTRS=" + str(dev_attrs) + "\n")
            if self.testset is not None:
                sys.stdout.write("\tevaluating on testset...")
                sys.stdout.flush()
                test_uas, test_las, test_upos, test_xpos, test_attrs = self.eval(self.testset)
                sys.stdout.write(
                    " UAS=" + str(test_uas) + " LAS=" + str(test_las) + " UPOS=" + str(test_upos) + " XPOS=" + str(
                        test_xpos) + " ATTRS=" + str(test_attrs) + "\n")

            if dev_uas > best_dev_uas:
                best_dev_uas = dev_uas
                dev_uas_uas = dev_uas
                dev_uas_las = dev_las
                if self.testset is not None:
                    test_uas_uas = test_uas
                    test_uas_las = test_las
                path = output_base + ".bestUAS"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.parser.save(path)
                itt_no_improve = self.patience

            if dev_las > best_dev_las:
                best_dev_las = dev_las
                dev_las_uas = dev_uas
                dev_las_las = dev_las
                if self.testset is not None:
                    test_las_uas = test_uas
                    test_las_las = test_las
                path = output_base + ".bestLAS"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.parser.save(path)
                itt_no_improve = self.patience

            path = output_base + ".last"
            sys.stdout.write("\tStoring " + path + "\n")
            sys.stdout.flush()
            self.parser.save(path)

        sys.stdout.write("Training is done with devset\n")
        sys.stdout.write("Best UAS score provides:\n")
        sys.stdout.write("\tDev UAS=" + str(dev_uas_uas) + " LAS=" + str(dev_uas_las) + "\n")
        sys.stdout.write("\tTest UAS=" + str(test_uas_uas) + " LAS=" + str(test_uas_las) + "\n")
        sys.stdout.write("Best LAS score provides:\n")
        sys.stdout.write("\tDev UAS=" + str(dev_las_uas) + " LAS=" + str(dev_las_las) + "\n")
        sys.stdout.write("\tTest UAS=" + str(test_las_uas) + " LAS=" + str(test_las_las) + "\n")
        sys.stdout.write("\n")

    def eval(self, dataset):
        last_proc = 0
        correct_uas = 0
        correct_las = 0
        correct_upos = 0
        correct_xpos = 0
        correct_attrs = 0

        total = 0
        for iSeq in range(len(dataset.sequences)):
            seq = dataset.sequences[iSeq]
            # remove compound words
            tmp = []
            for entry in seq:
                if not entry.is_compound_entry:
                    tmp.append(entry)
            seq = tmp
            proc = int((iSeq + 1) * 100 / len(dataset.sequences))
            if proc % 5 == 0 and proc != last_proc:
                last_proc = proc
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()

            predicted = self.parser.tag(seq)

            for entry, pred in zip(seq, predicted):
                total += 1
                gold_head = entry.head
                gold_label = entry.label
                pred_head = pred.head
                pred_label = pred.label

                if pred_head == gold_head:
                    correct_uas += 1
                    if gold_label == pred_label:
                        correct_las += 1

                if pred.upos == entry.upos:
                    correct_upos += 1
                if pred.xpos == entry.xpos:
                    correct_xpos += 1
                if pred.attrs == entry.attrs:
                    correct_attrs += 1

        if total == 0:
            total += 1
        return float(correct_uas) / total, float(correct_las) / total, float(correct_upos) / total, float(
            correct_xpos) / total, float(correct_attrs) / total


class TokenizerTrainer:
    def __init__(self, tokenizer, encodings, patience, trainset, devset=None, testset=None, raw_train_file=None,
                 raw_dev_file=None, raw_test_file=None, gold_train_file=None, gold_dev_file=None, gold_test_file=None):
        self.tokenizer = tokenizer
        self.trainset = trainset
        self.devset = devset
        self.raw_train_file = raw_train_file
        self.raw_dev_file = raw_dev_file
        self.raw_test_file = raw_test_file
        self.gold_train_file = gold_train_file
        self.gold_dev_file = gold_dev_file
        self.gold_test_file = gold_test_file
        self.testset = testset
        self.patience = patience
        self.encodings = encodings

    # creates from a Dataset.sequences (which is a list of lists of Conll entries)
    # a two lists of lists of chars and gold labels
    def _create_Xy_sequences(self, sequence_set):
        X_set = []
        y_set = []
        space_after_end_of_sentence = True
        space_after_end_of_sentence_count = 0
        for i in range(len(sequence_set.sequences)):  # for all sequences (sentences)
            X = []
            y = []
            # print(">> Starting sequence "+str(i))
            for j in range(len(sequence_set.sequences[i])):  # for each word in the sentence
                if sequence_set.sequences[i][j].is_compound_entry:  # skip over compound words
                    pass
                word = sequence_set.sequences[i][j].word
                space_after = False if "SpaceAfter=No" in sequence_set.sequences[i][j].space_after else True
                if sys.version_info[0] == 2:
                    uniword = unicode(word, 'utf-8')
                else:
                    import copy
                    uniword = copy.deepcopy(word)
                # print("  WORD: "+uniword+" len = "+str(len(uniword))+" space after = "+str(sequence_set.sequences[i][j].space_after))
                # mark all symbols as "O"
                for char_index in range(len(uniword)):
                    X.append(uniword[char_index])
                    y.append("O")
                    # last symbol always is "S"
                y[-1] = "S"
                # is this the last symbol in the sentence? 
                if j == len(sequence_set.sequences[i]) - 1:  # yes, last symbol in sentence, add an "X" to its label
                    y[-1] = "SX"
                    # permanently set space_after_end_of_sentence to False if we see just one sentence that does not have a space after
                    if not space_after:
                        space_after_end_of_sentence_count += 1
                else:  # add space after only if not end of sentence
                    if space_after:
                        X.append(" ")
                        y.append("S")
                        # for q in range(len(X)):
            #    print X[q]+" "+y[q]
            # raw_input("Stop")
            X_set.append(X)
            y_set.append(y)

        # if at least 75% of sentences end with space_after="No" then we assume this language does not use spaces after EOSes.
        if float(space_after_end_of_sentence_count) > 0.75 * len(sequence_set.sequences):
            space_after_end_of_sentence = False

        return X_set, y_set, space_after_end_of_sentence

    def _create_mixed_sequences(self, X_set, y_set, space_after_end_of_sentence, shuffle=False):
        assert (len(X_set) == len(y_set))
        # print(" Set has "+str(len(X_set))+" sequences")
        X_mixed_set = []
        y_mixed_set = []
        for i in range(len(X_set)):
            import copy
            X_mixed = copy.deepcopy(X_set[i])
            y_mixed = copy.deepcopy(y_set[i])
            if space_after_end_of_sentence:
                X_mixed.append(" ")
                y_mixed.append("S")
                # now, add random sentence
            pick = random.randint(0, len(X_set) - 1)
            X_mixed = X_mixed + copy.deepcopy(X_set[pick])
            y_mixed = y_mixed + copy.deepcopy(y_set[pick])
            if space_after_end_of_sentence:
                X_mixed.append(" ")
                y_mixed.append("S")
                # add some random chars of another sentence
            while True:  # some sequences have only one word, skip them
                pick = random.randint(0, len(X_set) - 1)
                if len(X_set[pick]) > 1:
                    break
            char_count = random.randint(1, len(X_set[pick]) - 1)
            X_mixed = X_mixed + copy.deepcopy(X_set[pick][0:char_count])
            y_mixed = y_mixed + copy.deepcopy(y_set[pick][0:char_count])

            # for q in range(len(X_mixed)):
            #    print X_mixed[q]+" "+y_mixed[q]
            # raw_input("Stop")
            X_mixed_set.append(X_mixed)
            y_mixed_set.append(y_mixed)

        if shuffle:
            combined = list(zip(X_mixed_set, y_mixed_set))
            random.shuffle(combined)
            X_mixed_set[:], y_mixed_set[:] = zip(*combined)

        return X_mixed_set, y_mixed_set

    def start_training(self, output_base, batch_size=0):
        epoch = 0
        itt_no_improve = self.patience
        best_dev_tok = 0.
        best_dev_ss = 0.
        selected_test_tok = 0.
        selected_test_ss = 0.
        path = output_base + ".encodings"
        sys.stdout.write("Storing encodings in " + path + "\n")
        self.encodings.save(path)
        path = output_base + ".conf"
        sys.stdout.write("Storing config in " + path + "\n")
        self.tokenizer.config.save(path)
        """
        sys.stdout.write("\tevaluating on devset...")
        sys.stdout.flush()
        dev_acc = self.eval(self.devset)
        sys.stdout.write(" accuracy=" + str(dev_acc) + "\n")
        if self.testset is not None:
            sys.stdout.write("\tevaluating on testset...")
            sys.stdout.flush()
            test_acc = self.eval(self.testset)
            sys.stdout.write(" accuracy=" + str(test_acc) + "\n")
        best_dev_acc = dev_acc
        """
        # convert Dataset to list of chars
        X_train_raw, y_train_raw, space_after_end_of_sentence = self._create_Xy_sequences(self.trainset)
        if not space_after_end_of_sentence:
            print ("\t NOTE: Training sentences do not end with a space after EOS.")
        # X_dev_raw, y_dev_raw, _ = self._create_Xy_sequences(self.devset)

        while itt_no_improve > 0:
            itt_no_improve -= 1
            epoch += 1
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.flush()
            sys.stdout.write("\tshuffling training data... ")
            sys.stdout.flush()
            X_train, y_train = self._create_mixed_sequences(X_train_raw, y_train_raw, space_after_end_of_sentence,
                                                            shuffle=True)
            sys.stdout.write("done\n")
            sys.stdout.flush()

            last_proc = 0
            sys.stdout.write("\ttraining...")
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()
            current_batch_size = 0
            self.tokenizer.start_batch()
            for iSeq in range(len(X_train)):
                # print("TRAIN SEQ: "+str(iSeq))
                X = X_train[iSeq]
                y = y_train[iSeq]
                current_batch_size += len(X)
                proc = int((iSeq + 1) * 100 / len(X_train))
                if proc % 5 == 0 and proc != last_proc:
                    last_proc = proc
                    sys.stdout.write(" " + str(proc))
                    sys.stdout.flush()

                self.tokenizer.learn_ss(X, y)
                self.tokenizer.learn_tok(X, y)
                if current_batch_size >= batch_size:
                    current_batch_size = 0
                    total_loss += self.tokenizer.end_batch()
                    self.tokenizer.start_batch()
            if current_batch_size != 0:
                current_batch_size = 0
                total_loss += self.tokenizer.end_batch()
                self.tokenizer.start_batch()

            stop_time = time.time()
            sys.stdout.write(" avg_loss=" + str(total_loss / len(X_train)) + " execution_time=" + str(
                stop_time - start_time) + "\n")

            # sys.stdout.write("\tevaluating on trainset...")
            # sys.stdout.flush()

            # train_tok, train_ss = self.eval(self.raw_train_file, self.gold_train_file)
            # train_tok, train_ss = self.eval(self.raw_test_file, self.gold_test_file)
            # sys.stdout.write(" token accuracy=" + str(train_tok) + " , sentence accuracy=" + str(train_ss) + "\n")

            sys.stdout.write("\tevaluating on devset...")
            sys.stdout.flush()
            dev_tok, dev_ss = self.eval(self.raw_dev_file, self.gold_dev_file)
            sys.stdout.write(" token_accuracy=" + str(dev_tok) + " , sentence_accuracy=" + str(dev_ss) + "\n")
            if self.testset is not None:
                sys.stdout.write("\tevaluating on testset...")
                sys.stdout.flush()
                test_tok, test_ss = self.eval(self.raw_test_file, self.gold_test_file)
                sys.stdout.write(" token_accuracy=" + str(test_tok) + " , sentence_accuracy=" + str(test_ss) + "\n")
            if dev_ss > best_dev_ss:
                best_dev_ss = dev_ss
                if self.testset is not None:
                    selected_test_ss = test_ss
                path = output_base + "-ss.bestAcc"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.tokenizer.save_ss(path)
                itt_no_improve = self.patience
            if dev_tok > best_dev_tok:
                best_dev_tok = dev_tok
                if self.testset is not None:
                    selected_test_tok = test_tok
                path = output_base + "-tok.bestAcc"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.tokenizer.save_tok(path)
                itt_no_improve = self.patience

            path = output_base + "-ss.last"
            sys.stdout.write("\tStoring " + path + "\n")
            sys.stdout.flush()
            self.tokenizer.save_ss(path)
            path = output_base + "-tok.last"
            sys.stdout.write("\tStoring " + path + "\n")
            sys.stdout.flush()
            self.tokenizer.save_ss(path)

        sys.stdout.write(
            "Training is done with devset sentence tok = " + str(best_dev_tok) + " and sentence = " + str(best_dev_ss))
        if self.testset is not None:
            sys.stdout.write(
                " and testset sentence tok = " + str(selected_test_tok) + " and sentence = " + str(selected_test_ss) +
                "(for the selected epoch, based on best devset tok/ss accuracy)")
        sys.stdout.write("\n")

    def eval(self, raw_text_file, gold_conllu_file):
        input_string = ""
        useSpaces = " "  # True
        lines = []

        with fopen(raw_text_file, "r") as file:
            lines = file.readlines()

        # analyze use of spaces in first part of the file
        test = "";
        cnt = 0
        while True:
            test = test + lines[cnt]
            # print(lines[cnt])
            cnt += 1
            if cnt >= len(lines) or cnt > 5:
                break

        if float(test.count(' ')) / float(len(test)) < 0.02:
            useSpaces = ""
        # print (str(float(test.count(' '))/float(len(test))))

        i = -1
        input_string = ""
        sentences = []
        while i < len(lines) - 1:
            i += 1
            input_string = input_string + lines[i].replace("\r", "").replace("\n", "").strip() + useSpaces
            if lines[i].strip() == "" or i == len(lines) - 1:  # end of block
                if input_string.strip() != "":
                    sentences += self.tokenizer.tokenize(input_string)
                input_string = ""

        with fopen(self.tokenizer.config.base + "-temporary.conllu", 'w') as file:
            for sentence in sentences:
                # print ("Sentence has entries: "+str(len(sentence)))
                for entry in sentence:
                    line = str(
                        entry.index) + "\t" + entry.word + "\t" + entry.lemma + "\t" + entry.upos + "\t" + entry.xpos + "\t" + entry.attrs + "\t" + str(
                        entry.head) + "\t" + entry.label + "\t" + entry.deps + "\t" + entry.space_after + "\n"
                    file.write(line)

                file.write("\n")

        # run eval script
        metrics = conll_eval(self.tokenizer.config.base + "-temporary.conllu", gold_conllu_file)

        return metrics["Tokens"].f1 * 100., metrics["Sentences"].f1 * 100.


class NERTrainer:
    def __init__(self, ner, encodings, patience, trainset, devset, testset=None):
        self.ner = ner
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.patience = patience
        self.encodings = encodings

    def start_training(self, output_base, batch_size=100):

        epoch = 0
        itt_no_improve = self.patience

        path = output_base + ".encodings"
        sys.stdout.write("Storing encodings in " + path + "\n")
        self.encodings.save(path)
        path = output_base + ".conf"
        sys.stdout.write("Storing config in " + path + "\n")
        self.ner.config.save(path)
        sys.stdout.write("\tevaluating on devset...")
        sys.stdout.flush()

        best_dev_score = 0
        best_dev_precision = 0
        best_dev_recall = 0
        best_test_score = 0
        best_test_precision = 0
        best_test_recall = 0

        current_batch_size = 0
        self.ner.start_batch()

        while itt_no_improve > 0:

            itt_no_improve -= 1
            epoch += 1
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.flush()
            sys.stdout.write("\tshuffling training data... ")
            sys.stdout.flush()
            shuffle(self.trainset.sequences)
            sys.stdout.write("done\n")
            sys.stdout.flush()
            last_proc = 0
            sys.stdout.write("\ttraining...")
            sys.stdout.flush()
            total_loss = 0
            start_time = time.time()

            for iSeq in range(len(self.trainset.sequences)):
                seq = self.trainset.sequences[iSeq]
                proc = int((iSeq + 1) * 100 / len(self.trainset.sequences))
                if proc % 5 == 0 and proc != last_proc:
                    while last_proc < proc:
                        last_proc += 5
                        sys.stdout.write(" " + str(last_proc))
                        sys.stdout.flush()

                self.ner.learn(seq)
                current_batch_size += len(seq)
                if current_batch_size >= batch_size:
                    total_loss += self.ner.end_batch()
                    current_batch_size = 0
                    self.ner.start_batch()
            total_loss += self.ner.end_batch()
            current_batch_size = 0
            stop_time = time.time()
            sys.stdout.write(" avg_loss=" + str(total_loss / len(self.trainset.sequences)) + " execution_time=" + str(
                stop_time - start_time) + "\n")
            self.ner.start_batch()

            sys.stdout.write("\tevaluating on devset...")
            sys.stdout.flush()
            dev_precision, dev_recall, dev_score = self.eval(self.devset)
            sys.stdout.write(
                " P=" + str(dev_precision) + " R=" + str(dev_recall) + " F=" + str(dev_score) + "\n")
            if self.testset is not None:
                sys.stdout.write("\tevaluating on testset...")
                sys.stdout.flush()
                test_precision, test_recall, test_score = self.eval(self.testset)
                sys.stdout.write(
                    " P=" + str(test_precision) + " R=" + str(test_recall) + " F=" + str(test_score) + "\n")

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                best_dev_precision = dev_precision
                best_dev_recall = dev_recall

                if self.testset is not None:
                    best_test_score = test_score
                    best_test_precision = test_precision
                    best_test_recall = test_recall
                path = output_base + ".bestFScore"
                sys.stdout.write("\tStoring " + path + "\n")
                sys.stdout.flush()
                self.ner.save(path)
                itt_no_improve = self.patience

            path = output_base + ".last"
            sys.stdout.write("\tStoring " + path + "\n")
            sys.stdout.flush()
            self.ner.save(path)

        sys.stdout.write("Training is done with devset\n")
        sys.stdout.write("Best UAS score provides:\n")
        sys.stdout.write(
            "\tDev P=" + str(best_dev_precision) + " R=" + str(best_dev_recall) + " F=" + str(best_dev_score) + "\n")
        if self.testset is not None:
            sys.stdout.write("\tTest P=" + str(best_test_precision) + " R=" + str(best_test_recall) + " F=" + str(
                best_test_score) + "\n")
        sys.stdout.write("\n")

    def eval(self, dataset):
        true_p = 0
        false_p = 0
        total_p = 0
        last_proc = 0
        iSeq = 0

        for s in dataset.sequences:
            iSeq += 1
            proc = int(iSeq * 100 / len(dataset.sequences))
            if proc % 15 == 0 and last_proc != proc:
                while last_proc < proc:
                    last_proc += 5
                    sys.stdout.write(" " + str(last_proc))
                    sys.stdout.flush()

            import dynet as dy
            dy.renew_cg()  # This is a special case for trainers. We evaluate the graph itself instead of the final output

            output, proj_x = self.ner._predict(s, runtime=True)
            for iSrc in range(len(s)):
                for iDst in range(len(s)):
                    if iDst > iSrc:
                        # from network import get_link
                        from generic_networks.ner import get_link
                        link = get_link(s, iSrc, iDst)
                        if link == 1:
                            total_p += 1
                        p_val = output[iSrc][iDst].value()
                        if p_val >= 0.5:
                            link_pred = 1
                        else:
                            link_pred = 0

                        if link_pred == 1 and link == 1:
                            true_p += 1
                        elif link_pred == 1:
                            false_p += 1

        # print(" ", true_p, false_p, total_p)
        if false_p + true_p == 0:
            false_p += 1
        precision = float(true_p) / float(false_p + true_p)
        if total_p == 0:
            total_p += 1
        recall = float(true_p) / total_p
        if precision == 0 or recall == 0:
            fscore = 0
        else:
            fscore = float(2 * precision * recall) / (precision + recall)

        return precision, recall, fscore
