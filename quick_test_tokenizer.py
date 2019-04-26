from cube.io_utils.conll import Dataset

trainset = Dataset()
devset = Dataset()

train_list = ['corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-train.conllu',
              'corpus/ud-treebanks-v2.2/UD_French-Sequoia/fr_sequoia-ud-train.conllu',
              'corpus/ud-treebanks-v2.2/UD_French-GSD/fr_gsd-ud-train.conllu',
              'corpus/ud-treebanks-v2.2/UD_Portuguese-Bosque/pt_bosque-ud-train.conllu',
              'corpus/ud-treebanks-v2.2/UD_Spanish-AnCora/es_ancora-ud-train.conllu',
              'corpus/ud-treebanks-v2.2/UD_Catalan-AnCora/ca_ancora-ud-train.conllu',
              'corpus/ud-treebanks-v2.2/UD_French-Spoken/fr_spoken-ud-train.conllu',
              'corpus/ud-treebanks-v2.2/UD_Galician-CTG/gl_ctg-ud-train.conllu',
              'corpus/ud-treebanks-v2.2/UD_Italian-ISDT/it_isdt-ud-train.conllu',
              'corpus/ud-treebanks-v2.2/UD_Italian-PoSTWITA/it_postwita-ud-train.conllu']

dev_list = ['corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-dev.conllu',
            'corpus/ud-treebanks-v2.2/UD_French-Sequoia/fr_sequoia-ud-dev.conllu',
            'corpus/ud-treebanks-v2.2/UD_French-GSD/fr_gsd-ud-dev.conllu',
            'corpus/ud-treebanks-v2.2/UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu',
            'corpus/ud-treebanks-v2.2/UD_Spanish-AnCora/es_ancora-ud-dev.conllu',
            'corpus/ud-treebanks-v2.2/UD_Catalan-AnCora/ca_ancora-ud-dev.conllu',
            'corpus/ud-treebanks-v2.2/UD_French-Spoken/fr_spoken-ud-dev.conllu',
            'corpus/ud-treebanks-v2.2/UD_Galician-CTG/gl_ctg-ud-dev.conllu',
            'corpus/ud-treebanks-v2.2/UD_Italian-ISDT/it_isdt-ud-dev.conllu',
            'corpus/ud-treebanks-v2.2/UD_Italian-PoSTWITA/it_postwita-ud-dev.conllu']

trainset = Dataset()
devset = Dataset()
for ii in range(len(train_list)):
    trainset.load_language(train_list[ii], ii)
    devset.load_language(dev_list[ii], ii)

# # trainset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-train.conllu', 0)
# trainset.load_language('corpus/ud-treebanks-v2.2/UD_Japanese-GSD/ja_gsd-ud-train.conllu', 0)
# trainset.load_language('corpus/ud-treebanks-v2.2/UD_Chinese-GSD/zh_gsd-ud-train.conllu', 1)
# trainset.load_language('corpus/ud-treebanks-v2.2/UD_Korean-GSD/ko_gsd-ud-train.conllu', 2)
# trainset.load_language('corpus/ud-treebanks-v2.2/UD_Korean-Kaist/ko_kaist-ud-train.conllu', 3)
# devset = Dataset()
# # devset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-dev.conllu', 0)
# devset.load_language('corpus/ud-treebanks-v2.2/UD_Japanese-GSD/ja_gsd-ud-dev.conllu', 0)
# devset.load_language('corpus/ud-treebanks-v2.2/UD_Chinese-GSD/zh_gsd-ud-dev.conllu', 1)
# devset.load_language('corpus/ud-treebanks-v2.2/UD_Korean-GSD/ko_gsd-ud-dev.conllu', 2)
# devset.load_language('corpus/ud-treebanks-v2.2/UD_Korean-Kaist/ko_kaist-ud-dev.conllu', 3)

from cube.io_utils.encodings import Encodings

encodings = Encodings()
encodings.compute(trainset, devset, char_cutoff=2)

from cube.generic_networks.tokenizers import CRFTokenizer
from cube.io_utils.config import TokenizerConfig

config = TokenizerConfig()
tokenizer = CRFTokenizer(config, encodings, num_languages=len(train_list))

from cube.io_utils.trainers import TokenizerTrainer

trainer = TokenizerTrainer(tokenizer, encodings, 20, trainset, devset)
trainer.start_training('tokenizer-mixed-romance', 1000)
