import sys
sys.path.append('')
import torch
from cube3.networks.tokenizer import Tokenizer
from cube3.io_utils.config import TokenizerConfig
from cube3.io_utils.encodings import Encodings
from cube3.networks.utils_tokenizer import TokenCollateFTLanguasito

enc=Encodings()
enc.load('data/tokenizer-ro-fasttext.encodings')

config=TokenizerConfig()
tokenizer=Tokenizer(config, enc, language_codes=['ro_nonstandard', 'ro_rrt'], ext_word_emb=300)

model=torch.load('data/tokenizer-ro-fasttext.ro_rrt.sent')

tokenizer.load_state_dict(model['state_dict'])

collate=TokenCollateFTLanguasito(enc, lm_model='fasttext:ro')
d=tokenizer.process('Ana are mere, dar nu are pere. Evenimentul are ca scop facilitarea schimbului de idei privind viitorul securitatii energetice in aceste regiuni. Nu-i treaba mea sa stiu asta.', collate, lang_id=1)


from ipdb import set_trace
set_trace()
