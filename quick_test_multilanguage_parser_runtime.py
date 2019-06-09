# this is a quick testing script for romance languages


from cube.io_utils.conll import Dataset

train_list = ['corpus/ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_Romanian-Nonstandard/ro_nonstandard-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_French-Sequoia/fr_sequoia-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_French-GSD/fr_gsd-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_Portuguese-Bosque/pt_bosque-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_Spanish-AnCora/es_ancora-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_Catalan-AnCora/ca_ancora-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_French-Spoken/fr_spoken-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_Galician-CTG/gl_ctg-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_Italian-ISDT/it_isdt-ud-train.conllu',
              'corpus/ud-treebanks-v2.4/UD_Italian-PoSTWITA/it_postwita-ud-train.conllu']

dev_list = ['corpus/ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_Romanian-Nonstandard/ro_nonstandard-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_French-Sequoia/fr_sequoia-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_French-GSD/fr_gsd-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_Spanish-AnCora/es_ancora-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_Catalan-AnCora/ca_ancora-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_French-Spoken/fr_spoken-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_Galician-CTG/gl_ctg-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_Italian-ISDT/it_isdt-ud-dev.conllu',
            'corpus/ud-treebanks-v2.4/UD_Italian-PoSTWITA/it_postwita-ud-dev.conllu']
train_list = train_list[:2]
dev_list = dev_list[:2]

trainset = Dataset()
devset = Dataset()
for ii in range(len(train_list)):
    trainset.load_language(train_list[ii], ii)
    devset.load_language(dev_list[ii], ii)

from cube.io_utils.trainers import ParserTrainer
from cube.io_utils.encodings import Encodings
from cube.generic_networks.parsers import BDRNNParser
from cube.io_utils.config import ParserConfig

encodings = Encodings()
# encodings.compute(trainset, devset, tag_type='label')
encodings.load('model-mixed-parser.encodings')
config = ParserConfig()
tagger = BDRNNParser(config, encodings, num_languages=len(train_list), runtime=True)
tagger.load('model-mixed-parser.bestUAS')
for ii in range(len(train_list)):
    print (dev_list[ii])
    testset=Dataset()
    testset.load_language(dev_list[ii].replace('-dev','-test'), ii)
    new_seqs=tagger.parse_sequences(testset.sequences)
    f=open(dev_list[ii].replace('-dev','-test.output'),'w')
    for seq in new_seqs:
        for entry in seq:
            f.write(str(entry.index)+"\t"+entry.word+"\t"+entry.lemma+"\t"+entry.upos+"\t"+entry.xpos+"\t"+entry.attrs+"\t"+str(entry.head)+"\t"+str(entry.label)+"\t"+str(entry.deps)+"\t"+entry.space_after+'\n')
        f.write('\n')
    f.close()
    #tagger.parse_sequences()

#trainer = ParserTrainer(tagger, encodings, 20, trainset, devset)
#trainer.start_training('model-mixed-parser', batch_size=1000)
