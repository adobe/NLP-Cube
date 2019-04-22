from cube.io_utils.conll import Dataset

trainset = Dataset()
trainset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-train.conllu', 0)
devset = Dataset()
devset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-dev.conllu', 0)

from cube.io_utils.encodings import Encodings

encodings = Encodings()
encodings.compute(trainset, devset)

from cube.generic_networks.tokenizers import CRFTokenizer
from cube.io_utils.config import TokenizerConfig

config = TokenizerConfig()
tokenizer = CRFTokenizer(config, encodings)

from cube.io_utils.trainers import TokenizerTrainer

trainer = TokenizerTrainer(tokenizer, encodings, 20, trainset, devset)
trainer.start_training('tokenizer', 1000)
