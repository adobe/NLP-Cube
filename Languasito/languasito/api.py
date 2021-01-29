import sys
import torch
from typing import *

sys.path.append('')

from languasito.model import Languasito
from languasito.utils import LanguasitoCollate
from languasito.utils import Encodings


class LanguasitoAPI:

    def __init__(self, languasito: Languasito, encodings: Encodings):
        self._languasito = languasito
        self._languasito.eval()
        self._encodings = encodings
        self._collate = LanguasitoCollate(encodings)

    def to(self, device: str):
        self._languasito.to(device)

    def __call__(self, batch):
        with torch.no_grad():
            x = LanguasitoCollate(batch)
        rez = self._languasito(x)
        emb = []
        pred_emb = rez['emb'].detach().cpu().numpy()
        for ii in range(len(batch)):
            c_emb = []
            for jj in range(len(batch[ii])):
                c_emb.append(pred_emb[ii, jj])
            emb.append(c_emb)
        return emb

    @staticmethod
    def load(model: str):
        from pathlib import Path
        home = str(Path.home())
        filename = '{0}/.languasito/{1}'.format(home, model)
        import os
        if os.path.exists(filename):
            return LanguasitoAPI.load_local(filename)
        else:
            print("UserWarning: Model not found and automatic downloading is not yet supported")
            return None

    @staticmethod
    def load_local(model: str):
        enc = Encodings()
        enc.load('{0}.encodings'.format(model))
        model = Languasito(enc)
        model.load('{0}.model'.format(model))
        api = LanguasitoAPI(model, enc)
        return api
