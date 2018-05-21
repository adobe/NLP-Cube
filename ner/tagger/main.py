# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.


import dynet_config

dynet_config.set(mem=2048, autobatch=False)
# dynet_config.set_gpu()

import sys
import time
import os

# Append parent dir to sys path.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parent_dir)

from tagger.config import Config
from tagger.dataset import Dataset, Encodings
from tagger.network import Network

import dynet as dy


def display_help():
    print ("Neural tagger version 0.9 beta.")
    print ("Usage:")
    print ("\t--train <train file> <dev file> <model output base> <num itt no improve> [config file]")
    print ("\t--split <train file> <dev file> <output_base>")
    print ("\t--test <model output base> <embeddings file> <test file> <output file>")


def eval_sequences(network, seq):
    true_p = 0
    false_p = 0
    total_p = 0
    last_proc = 0
    iSeq = 0

    for s in seq:
        iSeq += 1
        proc = iSeq * 100 / len(seq)
        if proc % 15 == 0 and last_proc != proc:
            sys.stdout.write(" " + str(proc))
            sys.stdout.flush()
            last_proc = proc

        dy.renew_cg()
        output, proj_x = network.predict(s)
        for iSrc in range(len(s)):
            for iDst in range(len(s)):
                if iDst > iSrc:
                    from network import get_link
                    link = get_link(s, iSrc, iDst)
                    if link == 1:
                        total_p += 1
                    p_val = output[iSrc][iDst].value()
                    if p_val[0] < p_val[1]:
                        link_pred = 1
                    else:
                        link_pred = 0

                    if link_pred == 1 and link == 1:
                        true_p += 1
                    elif link_pred == 1:
                        false_p += 1

    print(" ", true_p, false_p, total_p)
    if false_p + true_p == 0:
        false_p += 1
    precision = float(true_p) / float(false_p + true_p)
    if total_p == 0:
        total_p += 1
    recall = float(true_p) / total_p
    if precision == 0 or recall == 0:
        f = 0
    else:
        f = float(2 * precision * recall) / (precision + recall)
    return precision, recall, f


def eval(network, dataset):
    sys.stdout.write(" T")
    sys.stdout.flush()
    sys.stdout.write(" D")
    sys.stdout.flush()
    d_prec, d_recall, d_f = eval_sequences(network, dataset.sequences)
    return d_prec, d_recall, d_f


def has_mwes(seq):
    return True
    for entry in seq:
        if entry.label != '*':
            return True
    return False


def do_split(train, dev, output_base):
    ds_train = Dataset(train)
    ds_dev = Dataset(dev)
    for seq in ds_dev.sequences:
        ds_train.sequences.append(seq)

    index = 0
    f_train = open(output_base + "-train.cupt", "w")
    f_dev = open(output_base + "-dev.cupt", "w")
    f = None
    for seq in ds_train.sequences:
        if has_mwes(seq):
            index += 1
            if index % 10 == 0:
                f = f_dev
            else:
                f = f_train

            for entry in seq[1:]:
                f.write(entry.orig_line + '\n')
            f.write("\n")
    f_train.close()
    f_dev.close()


def do_train(train, dev, output_base, itt_no_improve, config):
    print "train='" + train + "'"
    print "dev='" + dev + "'"
    ds_train = Dataset(train)
    ds_dev = Dataset(dev)
    encodings = Encodings(ds_train)
    encodings.store(output_base + '.encodings')
    # sys.exit(0)
    network = Network(config, encodings)
    epoch = 0
    itt = itt_no_improve
    # d_p, d_r, d_f = eval(network, ds_dev)
    # sys.stdout.write("P, R, F dev(" + str(d_p) + ", " + str(d_r) + ", " + str(d_f) + ")" + "\n")
    best_fscore = 0
    batch_size = 1000
    current_batch_size = 0
    network.start_batch()
    while itt > 0:
        total_loss = 0
        last_proc = 0
        sys.stdout.write("Epoch " + str(epoch))
        last_proc = 0
        for iSeq in range(len(ds_train.sequences)):
            if has_mwes(ds_train.sequences[iSeq]):
                proc = iSeq * 100 / len(ds_train.sequences)
                if proc != last_proc and proc % 5 == 0:
                    last_proc = proc
                    sys.stdout.write(" " + str(proc))
                    sys.stdout.flush()
                network.learn(ds_train.sequences[iSeq])
                current_batch_size += len(ds_train.sequences[iSeq])
                if current_batch_size >= batch_size:
                    current_batch_size = 0
                    total_loss += network.end_batch()
                    network.start_batch()

        if current_batch_size != 0:
            current_batch_size = 0
            total_loss += network.end_batch()
            network.start_batch()

        d_p, d_r, d_f = eval(network, ds_dev)
        sys.stdout.write(" average train loss=" + str(total_loss / len(ds_train.sequences)) + " P, R, F dev(" + str(
            d_p) + ", " + str(d_r) + ", " + str(d_f) + ")" + "\n")
        sys.stdout.flush()
        if d_f > best_fscore:
            network.save_network(output_base + "-best-fscore.network")
            best_fscore = d_f
            itt = itt_no_improve
        network.save_network(output_base + "-last.network")
        itt -= 1
        epoch += 1
    print("\n\n Training done: best F-SCORE is", best_fscore)


def test(model_output_base, test_file, output_file):
    config = Config()
    ds = Dataset(test_file)
    # ds.restore_encodings(model_output_base)
    encodings = Encodings(None)
    encodings.load(model_output_base + ".encodings")
    network = Network(config, encodings)
    network.load_network(model_output_base + "-best-fscore.network")
    f = open(output_file, "w")
    f.write('# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE\n')
    last_proc = 0
    for iSeq in range(len(ds.sequences)):
        seq = ds.sequences[iSeq]
        proc = iSeq * 100 / len(ds.sequences)
        if proc != last_proc and proc % 5 == 0:
            last_proc = proc
            sys.stdout.write(" " + str(proc))
            sys.stdout.flush()
        # output = matricea de adiacenta (primii 2 vectori)
        # proj_x = 3rd vector
        output, proj_x = network.predict(seq)
        #from ipdb import set_trace
        #set_trace()
        # expressions - lista de liste (indexes of words from expressions)
        # labels - label corespondent expresiilor
        # len(expressions) == len(labels)
        expressions, labels = network.decode(output, proj_x)

        # output=[["","","",""]*len(seq)]
        output = []
        [output.append([''] * 11) for _ in range(len(seq))]
        # print output
        for iRow in range(len(seq)):
            parts = seq[iRow].orig_line.split("\t")
            for iCol in range(10):
                output[iRow][iCol] = parts[iCol]

        for expr, label, expr_index in zip(expressions, labels, range(len(labels))):
            if output[expr[0]][10] != "":
                output[expr[0]][10] += ";"
            output[expr[0]][10] += str(expr_index + 1) + ":" + label
            for index in expr[1:]:
                if output[index][10] != "":
                    output[index][10] += ";"
                output[index][10] += str(expr_index + 1)

        for line in output:
            if line[-1] == "":
                line[-1] = "*"

        for line, orig in zip(output[1:], seq[1:]):
            for w in line[:-1]:
                f.write(w + "\t")
            # f.write(orig.label + '\t')
            f.write(line[-1] + "\n")

        f.write("\n")
    sys.stdout.write(' done\n')

    f.close()


if len(sys.argv) == 1:
    display_help()
else:
    if (sys.argv[1] == "--train" and (len(sys.argv) == 6) or (len(sys.argv) == 7)):
        config = Config()
        if len(sys.argv) == 7:
            config = sys.argv[6]
        do_train(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), config)
    elif (sys.argv[1] == "--split" and len(sys.argv) == 5):
        do_split(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        if (sys.argv[1] == "--test" and len(sys.argv) == 5):
            test(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            display_help()
