import sys

sys.path.append('')
import torch

from languasito.utils import Encodings, LanguasitoCollate
from languasito.model import Languasito

enc = Encodings()  # filename='data/tokenizer-ro-fasttext')

enc.load('data/laro.encodings')
checkpoint = torch.load('data/laro.best', map_location='cpu')
model = Languasito(encodings=enc)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
collate = LanguasitoCollate(enc, live=True)
text = ['Am citit despre pancreas și sucul pancreatic .'.split(' '), 'Pancreasul secretă suc pancreatic .'.split(' '),
        'Ana are mere dar nu are pere'.split(' '),
        'Steagul României , de asemenea cunoscut ca drapelul , are culorile albastru , galben și roșu .'.split(' ')]

batch = collate.collate_fn(text)
y = model(batch, return_w=True, imagine=True)


def _get_word(w_emb):
    word = ''
    for ii in range(w_emb.shape[0]):
        c_idx = w_emb[ii]
        if c_idx == 3:
            break
        else:
            word += enc.char_list[c_idx]
    return word


w_emb = y['x_char_pred']
print(w_emb.shape)
for w in w_emb:
    print('"{0}"'.format(_get_word(w)))
