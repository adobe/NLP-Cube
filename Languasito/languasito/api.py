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
        self._collate = LanguasitoCollate(encodings, live=True)
        self._device = 'cpu'

    def to(self, device: str):
        self._languasito.to(device)
        self._device = device

    def __call__(self, batch):
        with torch.no_grad():
            x = self._collate.collate_fn(batch)
            for key in x:
                if isinstance(x[key], torch.Tensor):
                    x[key] = x[key].to(self._device)
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
    def load(model_name: str):
        from pathlib import Path
        home = str(Path.home())
        filename = '{0}/.languasito/{1}'.format(home, model_name)
        import os
        if os.path.exists(filename + '.encodings'):
            return LanguasitoAPI.load_local(filename)
        else:
            print("UserWarning: Model not found and automatic downloading is not yet supported")
            return None

    @staticmethod
    def load_local(model_name: str):
        enc = Encodings()
        enc.load('{0}.encodings'.format(model_name))
        model = Languasito(enc)
        tmp = torch.load('{0}.best'.format(model_name), map_location='cpu')
        # model.load(tmp['state_dict'])
        model.load_state_dict(tmp['state_dict'])
        model.eval()
        api = LanguasitoAPI(model, enc)
        return api
