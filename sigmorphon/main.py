import dynet_config
import optparse
import sys
import os
import copy
import time

from random import shuffle


def eval(lemmatizer, dataset, log_file=None):
    last_proc = 0
    correct = 0
    total = 0

    # from ipdb import set_trace
    # set_trace()

    f = None
    if log_file is not None:
        f = open(log_file, "w")

    for iSeq in xrange(len(dataset.sequences)):
        seq = dataset.sequences[iSeq]

        proc = (iSeq + 1) * 100 / len(dataset.sequences)
        if proc % 5 == 0 and proc != last_proc:
            last_proc = proc
            sys.stdout.write(" " + str(proc))
            sys.stdout.flush()

        pred_lemmas = lemmatizer.tag(seq)

        for entry, pred_lemma in zip(seq, pred_lemmas):

            if log_file is not None:
                f.write(entry.word + '\t' + entry.upos + '\t' + entry.lemma + '\t' + pred_lemma.encode('utf-8'))

            total += 1
            # from pdb import set_trace
            # set_trace()
            if unicode(entry.lemma, 'utf-8') == pred_lemma:
                correct += 1
            else:
                f.write('\t*')
            f.write('\n')

    if f is not None:
        f.close()
    return float(correct) / total


def train(train_file, dev_file, model_base, patience):
    from io_utils.sigmorphon import Sigmorphon2CONLL
    from io_utils.conll import Dataset

    ds_train = Sigmorphon2CONLL()
    ds_train.read_from_file(train_file)
    ds_train = ds_train.convert2conll()

    ds_dev = Sigmorphon2CONLL()
    ds_dev.read_from_file(dev_file)
    ds_dev = ds_dev.convert2conll()

    sys.stdout.write("Train file has " + str(len(ds_train.sequences)) + " sequences\n")
    sys.stdout.write("Dev file has " + str(len(ds_dev.sequences)) + " sequences\n")

    from io_utils.encodings import Encodings
    encodings = Encodings()
    encodings.compute(ds_train, ds_dev)
    sys.stdout.write("Storing encodings in " + model_base + ".encodings\n")
    encodings.save(model_base + ".encodings")

    num_itt_no_improve = patience
    best_dev_acc = 0

    from models.lemmatizers import FSTLemmatizer
    from models.lemmatizers import BDRNNLemmatizer
    from models.config import LemmatizerConfig

    config = LemmatizerConfig()

    config.save(model_base + ".config")

    lemmatizer = FSTLemmatizer(config, encodings, None, runtime=False)
    epoch = 0
    batch_size = 10
    while num_itt_no_improve > 0:

        epoch += 1
        sys.stdout.write("Starting epoch " + str(epoch) + "\n")
        sys.stdout.flush()
        sys.stdout.write("\tshuffling training data... ")
        sys.stdout.flush()

        shuffle(ds_train.sequences)
        sys.stdout.write("done\n")
        sys.stdout.flush()
        last_proc = 0
        sys.stdout.write("\ttraining...")
        sys.stdout.flush()
        total_loss = 0
        start_time = time.time()
        current_batch_size = 0
        lemmatizer.start_batch()
        for iSeq in xrange(len(ds_train.sequences)):
            seq = ds_train.sequences[iSeq]
            proc = (iSeq + 1) * 100 / len(ds_train.sequences)
            if proc % 5 == 0 and proc != last_proc:
                last_proc = proc
                sys.stdout.write(" " + str(proc))
                sys.stdout.flush()

            lemmatizer.learn(seq)
            current_batch_size += len(seq)
            if current_batch_size >= batch_size:
                total_loss += lemmatizer.end_batch()
                lemmatizer.start_batch()
                current_batch_size = 0
        total_loss += lemmatizer.end_batch()

        stop_time = time.time()
        sys.stdout.write(" avg_loss=" + str(total_loss / len(ds_train.sequences)) + " execution_time=" + str(
            stop_time - start_time) + "\n")
        sys.stdout.write("\tevaluating")
        dev_acc = eval(lemmatizer, ds_dev, model_base + ".log")
        sys.stdout.write(" devset accuracy is " + str(dev_acc) + "\n")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            lemmatizer.save(model_base + ".bestAcc")
            num_itt_no_improve = patience
        lemmatizer.save(model_base + ".last")
        num_itt_no_improve -= 1


if __name__ == '__main__':
    memory = int(2048)

    autobatch = False
    dynet_config.set(mem=memory, random_seed=9, autobatch=autobatch)
    import dynet as dy

    if sys.argv[1] == "--train":
        train(sys.argv[2], sys.argv[3], sys.argv[4], 200)
