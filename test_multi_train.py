# this is a quick testing script for romance languages


from cube.io_utils.conll import Dataset

test_list_out = ['../corpus/ud-treebanks-v2.2-test/UD_Romanian-RRT/ro_rrt-ud-test.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_French-Sequoia/fr_sequoia-ud-test.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_French-GSD/fr_gsd-ud-test.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_Spanish-AnCora/es_ancora-ud-test.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_Catalan-AnCora/ca_ancora-ud-test.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_French-Spoken/fr_spoken-ud-test.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_Galician-CTG/gl_ctg-ud-test.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_Italian-ISDT/it_isdt-ud-test.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_Italian-PoSTWITA/it_postwita-ud-test.conllu']

test_list = ['../tibimixedmodel/UD_Romanian-RRT.ro.ro.1.1.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_French-Sequoia/fr_sequoia-ud-test.conllu',
             '../tibimixedmodel/UD_French-GSD.fr.fr.1.1.conllu',
             '../tibimixedmodel/UD_Portuguese-Bosque.pt.pt.1.1.conllu',
             '../tibimixedmodel/UD_Spanish-AnCora.es.es.1.1.conllu',
             '../tibimixedmodel/UD_Catalan-AnCora.ca.ca.1.1.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_French-Spoken/fr_spoken-ud-test.conllu',
             '../tibimixedmodel/UD_Galician-CTG.gl.gl.1.1.conllu',
             '../tibimixedmodel/UD_Italian-ISDT.it.it.1.1.conllu',
             '../corpus/ud-treebanks-v2.2-test/UD_Italian-PoSTWITA/it_postwita-ud-test.conllu']

testset = Dataset()

from cube.io_utils.trainers import TaggerTrainer
from cube.io_utils.encodings import Encodings
from cube.generic_networks.taggers import BDRNNTagger
from cube.io_utils.config import TaggerConfig

encodings = Encodings()
encodings.load('../model-mixed.encodings')
config = TaggerConfig()
tagger = BDRNNTagger(config, encodings, num_languages=len(test_list), runtime=True)
tagger.load('../model-mixed.bestUPOS')

# trainer = TaggerTrainer(tagger, encodings, 20, trainset, devset)
# trainer.start_training('../model-mixed', batch_size=1000)


for ii in range(len(test_list)):
    testset = Dataset()
    testset.load_language(test_list[ii], ii)
    f = open(test_list[ii] + '.out', 'w')
    new_seqs = tagger.tag_sequences(testset.sequences)
    for seq in new_seqs:
        for entry in seq:
            f.write(str(
                entry.index) + "\t" + entry.word + "\t" + entry.lemma + "\t" + entry.upos + "\t" + entry.xpos + "\t" + entry.attrs + "\t" + str(
                entry.head) + "\t_\t_\t_\n")
        f.write("\n")
        import os

    f.close()
    os.system("python3 cube/misc/conll18_ud_eval.py --verbose " + test_list_out[ii] + " " + test_list[ii] + ".out")
