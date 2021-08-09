#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys
import os
import yaml
import string
import requests
import tarfile
from tqdm import tqdm

sys.path.append('')
from typing import Optional, Union
from cube.io_utils.objects import Document, Word, Token, Sentence
from cube.io_utils.encodings import Encodings
from cube.io_utils.config import CompoundConfig, TokenizerConfig, ParserConfig, LemmatizerConfig
from cube.networks.compound import Compound
from cube.networks.parser import Parser
from cube.networks.tokenizer import Tokenizer
from cube.networks.lemmatizer import Lemmatizer
from pathlib import Path
from cube.networks.lm import LMHelperHF, LMHelperFT
from cube.networks.utils_tokenizer import TokenCollateHF, TokenCollateFTLanguasito
from cube.networks.utils import MorphoCollate, Word2TargetCollate


class CubeObj:
    def __init__(self, model_base: str, device: str = 'cpu', lang: str = None):
        self._cwe = None
        # word expander
        path = '{0}-trf-cwe'.format(model_base)
        if os.path.exists('{0}.best'.format(path)):
            config = CompoundConfig(filename='{0}.config'.format(path))
            encodings = Encodings()
            encodings.load('{0}.encodings'.format(path))
            self._cwe = Compound(config, encodings)
            self._cwe.load('{0}.best'.format(path))
            self._cwe.to(device)

        # tokenizer
        path = '{0}-trf-tokenizer'.format(model_base)
        g_conf = yaml.safe_load(open('{0}.yaml'.format(path)))
        self._lang2id = {}
        for lng in g_conf['language_codes']:
            self._lang2id[lng] = len(self._lang2id)
        self._default_lang_id = self._lang2id[g_conf['language_map'][lang]]
        self._default_lang = lang
        config = TokenizerConfig(filename='{0}.config'.format(path))
        lm_model = config.lm_model
        encodings = Encodings()
        encodings.load('{0}.encodings'.format(path))
        if lm_model.startswith('transformer'):
            self._tokenizer_collate = TokenCollateHF(encodings,
                                                     lm_device=device,
                                                     lm_model=lm_model.split(':')[-1],
                                                     no_space_lang=config.no_space_lang,
                                                     lang_id=self._default_lang_id)
        else:
            self._tokenizer_collate = TokenCollateFTLanguasito(encodings,
                                                               lm_device=device,
                                                               lm_model=lm_model,
                                                               no_space_lang=config.no_space_lang,
                                                               lang_id=self._default_lang_id)


        self._tokenizer = Tokenizer(config, encodings, language_codes=g_conf['language_codes'],
                                    ext_word_emb=self._tokenizer_collate.get_embeddings_size())
        self._tokenizer.load('{0}.best'.format(path))
        self._tokenizer.to(device)

        # lemmatizer
        path = '{0}-trf-lemmatizer'.format(model_base)
        config = LemmatizerConfig(filename='{0}.config'.format(path))
        encodings = Encodings()
        encodings.load('{0}.encodings'.format(path))
        self._lemmatizer = Lemmatizer(config, encodings)
        self._lemmatizer.load('{0}.best'.format(path))
        self._lemmatizer.to(device)
        self._lemmatizer_collate = Word2TargetCollate(encodings)
        # parser-tagger
        path = '{0}-trf-parser'.format(model_base)
        config = ParserConfig(filename='{0}.config'.format(path))
        lm_model = config.lm_model
        if lm_model.startswith('transformer'):
            self._lm_helper = LMHelperHF(model=lm_model.split(':')[-1])
        else:
            self._lm_helper = LMHelperFT(model=lm_model.split(':')[-1])

        encodings = Encodings()
        encodings.load('{0}.encodings'.format(path))
        self._parser = Parser(config, encodings, language_codes=g_conf['language_codes'],
                              ext_word_emb=self._lm_helper.get_embedding_size())
        self._parser.load('{0}.best'.format(path))
        self._parser.to(device)
        self._parser_collate = MorphoCollate(encodings)

    def __call__(self, text: Union[str, Document], flavour: Optional[str] = None):
        lang_id = self._default_lang_id
        if flavour is not None:
            if flavour not in self._lang2id:
                print("Unsupported language flavour")
                print("Please choose from: {0}".format(' '.join([k for k in self._lang2id])))
                raise Exception("Unsupported language flavour\nPlease choose from: {0}".
                                format(' '.join([k for k in self._lang2id])))
            lang_id = self._lang2id[flavour]
        if isinstance(text, str):
            doc = self._tokenizer.process(text, self._tokenizer_collate, lang_id=lang_id, num_workers=0)
            if self._cwe is not None:
                doc = self._cwe.process(doc, self._lemmatizer_collate, num_workers=0)
        else:
            doc = text

        self._lm_helper.apply(doc)
        self._parser.process(doc, self._parser_collate, num_workers=0)
        self._lemmatizer.process(doc, self._lemmatizer_collate, num_workers=0)
        return doc


def _download_file(url: str, filename: str, description=None):
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception(f"Error getting {url}, received status_code {r.status_code}")
    file_size = int(r.headers['Content-Length'])
    chunk_size = 1024

    with open(filename, 'wb') as fp:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=description, unit_divisor=1024,
                  disable=True if description is None else False, leave=False) as progressbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk is not None:
                    fp.write(chunk)
                    fp.flush()
                    progressbar.update(len(chunk))

    return r.status_code


def _download_model(local_path, lang):
    download_base = "https://github.com/adobe/NLP-Cube-Models/raw/3.0/models/{0}.tar.gz-a".format(lang)
    file_base = "{0}.tar.gz-a".format(lang)
    terminations = string.ascii_lowercase[:20]
    file_list = []
    for t in terminations:
        download_url = '{0}{1}'.format(download_base, t)
        target_file = str(os.path.join(local_path, file_base))
        target_file = '{0}{1}'.format(target_file, t)
        try:
            if _download_file(download_url, target_file, description='Part {0}'.format(t)) != 200:
                break
        except:
            break
        file_list.append(target_file)

    target_file = os.path.join(local_path, file_base[:-2])

    f_out = open(target_file, 'wb')
    for file in file_list:
        f_in = open(file, 'rb')
        while True:
            buffer = f_in.read(1024 * 1024)
            if not buffer:
                break
            f_out.write(buffer)
    f_out.close()

    tar = tarfile.open(target_file, 'r:gz')
    tar.extractall(local_path)
    tar.close()


def _load(lang: str, device: Optional[str] = 'cpu') -> CubeObj:
    try:
        local_user_home = str(Path.home())
        local_user_storage = os.path.join(local_user_home, '.nlpcube', '3.0')
        os.makedirs(local_user_storage, exist_ok=True)
        lang_path = os.path.join(local_user_storage, lang)
        if not os.path.exists(lang_path):
            _download_model(local_user_storage, lang)

        return CubeObj('{0}/{1}'.format(lang_path, lang), device=device, lang=lang)
    except:
        raise Exception("There was a problem retrieving this language. Either it is unsupported or your Internet "
                        "connection is down.\n\nTo check for supported languages, visit "
                        "https://github.com/adobe/NLP-Cube/\n\nIt is hard to maintain models for all UD Treebanks. "
                        "This is way we are only including a handful of"
                        "languages with the official distribution. "
                        "However, we can include additional languages upon request"
                        "\n\nTo make a request for supporting a new language please create an issue on GitHub")


class Cube:
    def __init__(self, verbose=False):
        self._instance = None

    def load(self, lang: str, device: Optional[str] = 'cpu'):
        self._instance = _load(lang, device)

    def __call__(self, text: Union[str, Document], flavour: Optional[str] = None):
        return self._instance(text, flavour=flavour)
