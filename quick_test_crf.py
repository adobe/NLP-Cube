from cube.io_utils.conll import Dataset

trainset = Dataset()
trainset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-train.conllu', 0)
devset = Dataset()
devset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-dev.conllu', 0)

from cube.io_utils.encodings import Encodings

encodings = Encodings()
encodings.compute(trainset, devset)

import dynet as dy

model = dy.Model()
trainer = dy.AdamTrainer(model)

from cube.generic_networks.character_embeddings import CharacterNetwork

cn = CharacterNetwork(100, encodings, embeddings_size=100, model=model)

from cube.generic_networks.crf import CRFLabeler

labeler = CRFLabeler(len(encodings.upos2int), 2, 200, 100, model)

lang_emb = model.add_lookup_parameters((1, 100))
import tqdm


def build_input(seq):
    out_list = []
    for entry in seq:
        out_list.append(cn.compute_embeddings(entry.word, language_embeddings=lang_emb[0])[0])

    return out_list


def evaluate():
    correct = 0
    total = 0
    for idx in tqdm.tqdm(range(len(devset.sequences))):
        dy.renew_cg()
        seq = devset.sequences[idx][0]
        inp = build_input(seq)
        labels = labeler.viterbi_decoding(labeler.build_tagging_graph(inp))[0]

        for label, entry in zip(labels, seq):
            if label == encodings.upos2int[entry.upos]:
                correct += 1
            total += 1
    return correct / total


def train():
    total_loss = 0
    for idx in tqdm.tqdm(range(len(trainset.sequences))):
        dy.renew_cg()
        seq = trainset.sequences[idx][0]
        inp = build_input(seq)
        tags = []
        for entry in seq:
            tags.append(encodings.upos2int[entry.upos])
        loss = labeler.viterbi_loss(labeler.build_tagging_graph(inp), tags)[0]
        total_loss += loss.value() / len(inp)
        loss.backward()
        trainer.update()
    print("loss=" + str(total_loss / len(devset.sequences)))


# print(evaluate())

patience = 20
patience_left = patience
best = 0
while patience_left > 0:
    train()
    patience_left -= 1
    score = evaluate()
    print("devset acc=" + str(score))
    if score > best:
        best = score
        patience_left = patience
