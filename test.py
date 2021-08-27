import sys

sys.path.append('')
import torch
from cube.networks.tokenizer import Tokenizer
from cube.networks.tagger import Tagger
from cube.io_utils.config import TokenizerConfig, TaggerConfig, ParserConfig
from cube.io_utils.encodings import Encodings
from cube.networks.utils_tokenizer import TokenCollateFTLanguasito, TokenCollateHF
from cube.networks.lm import LMHelperFT, LMHelperHF
from cube.networks.utils import MorphoCollate
from cube.networks.parser import Parser

# # tokenizer
# enc = Encodings()
# enc.load('data/tokenizer-ro-transformer.encodings')
#
# config = TokenizerConfig()
# tokenizer = Tokenizer(config, enc, language_codes=['ro_nonstandard', 'ro_rrt'], ext_word_emb=[768 for _ in range(13)])
#
# model = torch.load('data/tokenizer-ro-transformer.ro_rrt.sent', map_location='cpu')
#
# tokenizer.load_state_dict(model['state_dict'])
#
# collate = TokenCollateHF(enc, lm_model='xlm-roberta-base', lm_device='cpu')
# text = open('corpus/ud-treebanks-v2.5/UD_Romanian-RRT/ro_rrt-ud-test.txt').read()
# # text = 'Și eu am mere. Ana are mere, dar nu are pere. Acesta este un test.'
# d = tokenizer.process(text, collate, lang_id=1, batch_size=4)
# tokenizer
enc = Encodings()
enc.load('data/tokenizer-ro-fasttext.encodings')

config = TokenizerConfig()
tokenizer = Tokenizer(config, enc, language_codes=['ro_rrt'], ext_word_emb=[300])

model = torch.load('data/tokenizer-ro-fasttext.ro_rrt.sent', map_location='cpu')

tokenizer.load_state_dict(model['state_dict'])

collate = TokenCollateFTLanguasito(enc, lm_model='fasttext:ro', lm_device='cpu')
text = open('corpus/ud-treebanks-v2.5/UD_Romanian-RRT/ro_rrt-ud-test.txt').read()
# text = 'Și eu am mere. Ana are mere, dar nu are pere. Acesta este un test.'
d = tokenizer.process(text, collate, lang_id=0, batch_size=4)
for ii in range(len(d.sentences)):
    d.sentences[ii].lang_id = 1

# helper = LMHelperFT(model='ro')
# helper.apply(d)

# # tagger
# enc = Encodings()
# enc.load('data/tagger-ro-fasttext.encodings')
# model = torch.load('data/tagger-ro-fasttext.ro_rrt.upos', map_location='cpu')
# config = TaggerConfig()
# config.load('data/tagger-ro-fasttext.config')
# tagger = Tagger(config, enc, ext_word_emb=helper.get_embedding_size(), language_codes=['ro_nonstandard', 'ro_rrt'])
# tagger.load_state_dict(model['state_dict'])
# collate = MorphoCollate(enc)
# d = tagger.process(d, collate)

# parser
# del helper
helper = LMHelperHF(model='xlm-roberta-base')
helper.apply(d)
enc = Encodings()
enc.load('data/parser-ro-transformer.encodings')
collate = MorphoCollate(enc)
model = torch.load('data/parser-ro-transformer.ro_rrt.uas', map_location='cpu')
config = ParserConfig()
config.load('data/parser-ro-transformer.config')
parser = Parser(config, enc, ext_word_emb=helper.get_embedding_size(), language_codes=['ro_nonstandard', 'ro_rrt'])
parser.load_state_dict(model['state_dict'])
d = parser.process(d, collate)

print(d)
print("")
