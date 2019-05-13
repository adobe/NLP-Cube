from cube.io_utils.conll import Dataset

trainset = Dataset()
trainset.load_language('corpus/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu', 0)
devset = Dataset()
devset.load_language('corpus/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu', 0)

from cube.io_utils.encodings import Encodings

encodings = Encodings()
encodings.compute(trainset, devset, word_cutoff=2, char_cutoff=3)

import dynet as dy

BATCH_SIZE = 1000

model = dy.Model()
trainer = dy.AdamTrainer(model, alpha=2e-3, beta_1=0.9, beta_2=0.9)

from cube.generic_networks.character_embeddings import CharacterNetwork

cn = CharacterNetwork(100, encodings, rnn_size=200, rnn_layers=2, embeddings_size=200, model=model,
                      lang_embeddings_size=1)

from cube.generic_networks.crf import CRFDecoder

labeler = dy.BiRNNBuilder(2, 200, 400, model, dy.LSTMBuilder)  # CRFLabeler(len(encodings.upos2int), 2, 200, 200, model)
decoder = CRFDecoder(model, 400, 300, len(encodings.upos2int))

lang_emb = model.add_lookup_parameters((1, 1))
word_emb = model.add_lookup_parameters((len(encodings.word2int), 200))
import tqdm


def build_input(seq, runtime=True):
    out_list = []
    for entry in seq:
        c_emb = cn.compute_embeddings(entry.word, language_embeddings=lang_emb[0], runtime=runtime)[0]
        w = entry.word.lower()
        if w in encodings.word2int:
            w_emb = word_emb[encodings.word2int[w]]
        else:
            w_emb = word_emb[encodings.word2int['<UNK>']]

        if runtime:
            emb = w_emb + c_emb
        else:
            import random
            p1 = random.random()
            p2 = random.random()
            mult = 1
            f1 = 1
            f2 = 1
            if p1 < 0.33:
                f1 = 0
                mult = 2
            if p2 < 0.33:
                f2 = 0
                mult = 2
            emb = (w_emb * f1 + c_emb * f2) * mult

        out_list.append(emb)

    return out_list


def evaluate():
    labeler.disable_dropout()
    correct_train = 0
    total_train = 0
    # for idx in tqdm.tqdm(range(len(trainset.sequences)), desc='\teval train', ncols=60):
    #     dy.renew_cg()
    #     seq = trainset.sequences[idx][0]
    #     inp = build_input(seq)
    #     labels = labeler.tag(inp)
    #
    #     for label, entry in zip(labels, seq):
    #         if entry.upos in encodings.upos2int and label == encodings.upos2int[entry.upos]:
    #             correct_train += 1
    #         total_train += 1
    total_train = 1
    correct_dev = 0
    total_dev = 0
    for idx in tqdm.tqdm(range(len(devset.sequences)), desc='\teval dev', ncols=60):
        dy.renew_cg()
        seq = devset.sequences[idx][0]
        inp = build_input(seq)
        # labels = labeler.tag(inp)
        enc = labeler.transduce(inp)
        labels = decoder.tag(enc)

        for label, entry in zip(labels, seq):
            if entry.upos in encodings.upos2int and label == encodings.upos2int[entry.upos]:
                correct_dev += 1
            total_dev += 1
    return correct_dev / total_dev, correct_train / total_train


def train():
    labeler.set_dropout(0.33)
    total_loss = 0
    losses = []
    in_batch = 0
    dy.renew_cg()
    total_samples = 0
    import random
    random.shuffle(trainset.sequences)
    for idx in tqdm.tqdm(range(len(trainset.sequences)), desc='\ttrainset', ncols=60):

        seq = trainset.sequences[idx][0]
        inp = build_input(seq, runtime=False)
        tags = []
        in_batch += len(inp)
        for entry in seq:
            tags.append(encodings.upos2int[entry.upos])
        # loss = labeler.learn(inp, tags)  # labeler.viterbi_loss(labeler.build_tagging_graph(inp), tags)[0]
        enc = labeler.transduce(inp)
        loss = decoder.learn(enc, tags)
        losses.append(loss)
        total_samples += len(inp)
        if in_batch > BATCH_SIZE:
            in_batch = 0
            loss = dy.esum(losses)
            total_loss += loss.value()
            loss.backward()
            trainer.update()
            losses = []
            dy.renew_cg()

    if len(losses) > 0:
        loss = dy.esum(losses)
        total_loss += loss.value()
        loss.backward()
        trainer.update()
        dy.renew_cg()

    print("loss=" + str(total_loss / total_samples))


print(evaluate())

patience = 20
patience_left = patience
best = 0
epoch = 1
while patience_left > 0:
    print("Starting epoch " + str(epoch))
    train()
    patience_left -= 1
    score_dev, score_train = evaluate()
    print("\tdevset acc=" + str(score_dev) + " trainser acc=" + str(score_train))
    if score_dev > best:
        best = score_dev
        patience_left = patience
        print("\tBest score yet, resetting patience")
    epoch += 1
