import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import *
import sys

sys.path.append('')
from cube.networks.modules import WordGram, LinearNorm
from cube.io_utils.encodings import Encodings
from cube.io_utils.config import DCWEConfig


class DCWE(pl.LightningModule):
    encodings: Encodings
    config: DCWEConfig

    def __init__(self, config: DCWEConfig, encodings: Encodings):
        super(DCWE, self).__init__()
        self._config = config
        self._encodings = encodings
        self._wg = WordGram(num_chars=len(encodings.char2int),
                            num_langs=encodings.num_langs,
                            num_layers=config.num_layers,
                            num_filters=config.num_filters,
                            char_emb_size=config.lang_emb_size,
                            case_emb_size=config.case_emb_size,
                            lang_emb_size=config.lang_emb_size
                            )
        self._output_proj = LinearNorm(config.num_filters // 2, config.output_size, w_init_gain='linear')
        self._improve = 0
        self._best_loss = 9999

    def forward(self, x_char, x_case, x_lang, x_mask, x_word_len):
        pre_proj = self._wg(x_char, x_case, x_lang, x_mask, x_word_len)
        proj = self._output_proj(pre_proj)
        return proj

    def _get_device(self):
        if self._output_proj.linear_layer.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._output_proj.linear_layer.weight.device.type,
                                str(self._output_proj.linear_layer.weight.device.index))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

    def training_step(self, batch, batch_idx):
        x_char = batch['x_char']
        x_case = batch['x_case']
        x_lang = batch['x_lang']
        x_word_len = batch['x_word_len']
        x_mask = batch['x_mask']
        y_target = batch['y_target']
        y_pred = self.forward(x_char, x_case, x_lang, x_mask, x_word_len)
        loss = torch.mean((y_pred - y_target) ** 2)
        return loss

    def validation_step(self, batch, batch_idx):
        x_char = batch['x_char']
        x_case = batch['x_case']
        x_lang = batch['x_lang']
        x_word_len = batch['x_word_len']
        x_mask = batch['x_mask']
        y_target = batch['y_target']
        y_pred = self.forward(x_char, x_case, x_lang, x_mask, x_word_len)
        loss = torch.mean((y_pred - y_target) ** 2)
        return {'loss': loss.detach().cpu().numpy()[0]}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        mean_loss = sum([output['loss'] for output in outputs])
        mean_loss /= len(outputs)
        self.log('val/loss', mean_loss)
        self.log('val/early_meta', self._improve)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, model_path: str, device: str = 'cpu'):
        self.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
        self.to(device)



