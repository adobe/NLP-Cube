import sys
sys.path.append('')
import torch
from cube3.networks.tokenizer import Tokenizer
from cube3.networks.tagger import Tagger
from cube3.io_utils.config import TokenizerConfig, TaggerConfig, ParserConfig
from cube3.io_utils.encodings import Encodings
from cube3.networks.utils_tokenizer import TokenCollateFTLanguasito
from cube3.networks.lm import LMHelperFT
from cube3.networks.utils import MorphoCollate
from cube3.networks.parser import Parser

# tokenizer
enc = Encodings()
enc.load('data/tokenizer-ro-fasttext.encodings')

config = TokenizerConfig()
tokenizer = Tokenizer(config, enc, language_codes=['ro_nonstandard', 'ro_rrt'], ext_word_emb=300)

model = torch.load('data/tokenizer-ro-fasttext.ro_rrt.sent', map_location='cpu')

tokenizer.load_state_dict(model['state_dict'])

collate = TokenCollateFTLanguasito(enc, lm_model='fasttext:ro')
text = open('corpus/ud-treebanks-v2.5/UD_Romanian-RRT/ro_rrt-ud-test.txt').read()
d = tokenizer.process(text, collate, lang_id=1, batch_size=4)

helper = LMHelperFT(model='ro')
helper.apply(d)

# tagger
enc = Encodings()
enc.load('data/tagger-ro-fasttext.encodings')
model = torch.load('data/tagger-ro-fasttext.ro_rrt.upos', map_location='cpu')
config = TaggerConfig()
config.load('data/tagger-ro-fasttext.config')
tagger = Tagger(config, enc, ext_word_emb=helper.get_embedding_size(), language_codes=['ro_nonstandard', 'ro_rrt'])
tagger.load_state_dict(model['state_dict'])
collate = MorphoCollate(enc)
d = tagger.process(d, collate)

# parser
enc = Encodings()
enc.load('data/parser-ro-fasttext.encodings')
model = torch.load('data/parser-ro-fasttext.ro_rrt.uas', map_location='cpu')
config = ParserConfig()
config.load('data/parser-ro-fasttext.config')
tagger = Parser(config, enc, ext_word_emb=helper.get_embedding_size(), language_codes=['ro_nonstandard', 'ro_rrt'])
tagger.load_state_dict(model['state_dict'])
d = tagger.process(d, collate)

print(d)
