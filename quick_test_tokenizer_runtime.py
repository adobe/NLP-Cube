from cube.io_utils.conll import Dataset

# trainset = Dataset()
# trainset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-train.conllu', 0)
# devset = Dataset()
# devset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-test.text', 0)

from cube.io_utils.encodings import Encodings

encodings = Encodings()
# encodings.compute(trainset, devset)
encodings.load('tokenizer-mixed-romance.encodings')

from cube.generic_networks.tokenizers import CRFTokenizer
from cube.io_utils.config import TokenizerConfig

test_list = ['corpus/ud-treebanks-v2.2/UD_Japanese-GSD/ja_gsd-ud-test.txt']
# test_list = ['corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-test.txt',
#             'corpus/ud-treebanks-v2.2/UD_French-Sequoia/fr_sequoia-ud-test.txt',
#             'corpus/ud-treebanks-v2.2/UD_French-GSD/fr_gsd-ud-test.txt',
#             'corpus/ud-treebanks-v2.2/UD_Portuguese-Bosque/pt_bosque-ud-test.txt',
#             'corpus/ud-treebanks-v2.2/UD_Spanish-AnCora/es_ancora-ud-test.txt',
#             'corpus/ud-treebanks-v2.2/UD_Catalan-AnCora/ca_ancora-ud-test.txt',
#             'corpus/ud-treebanks-v2.2/UD_French-Spoken/fr_spoken-ud-test.txt',
#             'corpus/ud-treebanks-v2.2/UD_Galician-CTG/gl_ctg-ud-test.txt',
#             'corpus/ud-treebanks-v2.2/UD_Italian-ISDT/it_isdt-ud-test.txt',
#             'corpus/ud-treebanks-v2.2/UD_Italian-PoSTWITA/it_postwita-ud-test.txt']

config = TokenizerConfig(filename='tokenizer-mixed-romance.conf')
tokenizer = CRFTokenizer(config, encodings, num_languages=10)
tokenizer.load('tokenizer-mixed-romance-tok.best')

for ii in range(len(test_list)):
    f = open(test_list[ii])
    text = f.read().replace('\r\n', ' ').replace('\n', ' ')
    while True:
        tt = text.replace('  ', ' ')
        if tt == text:
            break
        text = tt

    text = text.replace(' ', '')

    seqs = tokenizer.tokenize(text, lang_id=ii)
    with open("test-temporary.conllu", 'w') as file:
        for sentence in seqs:
            # print ("Sentence has entries: "+str(len(sentence)))
            for entry in sentence:
                line = str(
                    entry.index) + "\t" + entry.word + "\t" + entry.lemma + "\t" + entry.upos + "\t" + entry.xpos + "\t" + entry.attrs + "\t" + str(
                    entry.head) + "\t" + entry.label + "\t" + entry.deps + "\t" + entry.space_after + "\n"
                file.write(line)

            file.write("\n")
        file.close()

    from cube.misc.conll18_ud_eval_wrapper import conll_eval

    metrics = conll_eval("test-temporary.conllu", test_list[ii].replace(".txt", ".conllu"))
    print(test_list[ii])
    print("sent=" + str(metrics["Sentences"].f1) + " tokens=" + str(metrics["Tokens"].f1) + " words=" + str(
        metrics["Words"].f1))
