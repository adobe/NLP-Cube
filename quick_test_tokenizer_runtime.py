from cube.io_utils.conll import Dataset

#trainset = Dataset()
#trainset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-train.conllu', 0)
#devset = Dataset()
#devset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-test.text', 0)

from cube.io_utils.encodings import Encodings

encodings = Encodings()
#encodings.compute(trainset, devset)
encodings.load('tokenizer.encodings')

from cube.generic_networks.tokenizers import CRFTokenizer
from cube.io_utils.config import TokenizerConfig

config = TokenizerConfig(filename='tokenizer.conf')
tokenizer = CRFTokenizer(config, encodings)
tokenizer.load('tokenizer-ss.best')

f=open('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-test.txt')
text=f.read().replace('\r\n',' ').replace('\n',' ')
while True:
    tt=text.replace('  ',' ')
    if tt==text:
        break
    text=tt

seqs=tokenizer.tokenize(text)
with open("test-temporary.conllu", 'w') as file:
    for sentence in seqs:
        # print ("Sentence has entries: "+str(len(sentence)))
        for entry in sentence:
            line = str(
                entry.index) + "\t" + entry.word + "\t" + entry.lemma + "\t" + entry.upos + "\t" + entry.xpos + "\t" + entry.attrs + "\t" + str(
                entry.head) + "\t" + entry.label + "\t" + entry.deps + "\t" + entry.space_after + "\n"
            file.write(line)

        file.write("\n")

from cube.misc.conll18_ud_eval_wrapper import conll_eval

metrics = conll_eval("test-temporary.conllu", "corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-test.conllu")

        # return metrics["Tokens"].f1 * 100., metrics["Sentences"].f1 * 100.
print("sent="+str(metrics["Sentences"].f1)+" tokens="+str(metrics["Tokens"].f1)+" words="+str(metrics["Words"].f1))


# from cube.io_utils.trainers import TokenizerTrainer
#
# trainer = TokenizerTrainer(tokenizer, encodings, 20, trainset, devset)
# trainer.start_training('tokenizer', 1000)
