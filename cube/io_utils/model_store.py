#
# Author: Ruxandra Burtica
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

from collections import defaultdict
import io
import os
import sys
from zipfile import ZipFile

import requests
import xmltodict

# Append parent dir to sys path.
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

from io_utils.encodings import Encodings
from io_utils.embeddings import WordEmbeddings
from io_utils.config import (TieredTokenizerConfig, CompoundWordConfig,
                     LemmatizerConfig, TaggerConfig, ParserConfig)
from generic_networks.tokenizers import TieredTokenizer
from generic_networks.token_expanders import CompoundWordExpander
from generic_networks.lemmatizers import FSTLemmatizer
from generic_networks.taggers import BDRNNTagger
from generic_networks.parsers import BDRNNParser


class PipelineComponents(object):
    TOKENIZER = 'tokenizer'
    COMPOUND = 'compound'
    TAGGER = 'tagger'
    PARSER = 'parser'
    LEMMATIZER = 'lemmatizer'


class ModelStore(object):
    """
    Abstraction layer for working with models.

    Usage example:
        model_store = ModelStore()

        # Load models for lang_code ro.
        model_store.load('ro')

        # Get latest versions.
        model_store.get_latest_versions()
    """

    MODELS_PATH_LOCAL = 'models'
    MODELS_PATH_CLOUD = 'https://nlpcube.blob.core.windows.net/models'
    MODELS_PATH_CLOUD_ALL = os.path.join(MODELS_PATH_CLOUD, '?restype=container&comp=list')

    EMBEDDINGS_NAME = 'wiki.{}.vec'
    FACEBOOK_EMBEDDINGS_URL = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/'
    FACEBOOK_EMBEDDINGS_LOCATION = 'corpus/'

    def __init__(self, disk_path=None, cloud_path=None):
        self.disk_path = disk_path or self.MODELS_PATH_LOCAL
        self.cloud_path = cloud_path or self.MODELS_PATH_CLOUD
        self.models = {}

    def load(self, lang_code, check_for_latest=True):
        """
        Contains logic for loading or downloading and loading models for the target language.

        Args:
            lang_code: Target language code.
                See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
            check_for_latest: Whether or not to get the latest version.
        """
        version_to_download = self.version_to_donwload(lang_code, check_for_latest)
        if version_to_download:
            self._download_models_version(lang_code, version_to_download)

        # Now we surely have the language models downloaded
        self._load(lang_code)

    def _load(self, lang_code):
        """
        Load models on the class.
        """
        sys.stdout.write('Loading models for {}\n'.format(lang_code))
        path_for_language = os.path.join(self.disk_path, lang_code)

        # 1. Load word embeddings.
        self.embeddings = WordEmbeddings()
        word_embeddings_for_language = 'wiki.{}.vec'.format(lang_code)
        self.embeddings.read_from_file(
            os.path.join(path_for_language, word_embeddings_for_language), None, full_load=False)

        # 2. Load tokenizer.
        if not os.path.isfile(os.path.join(path_for_language, 'tokenizer-tok.bestAcc')):
            sys.stdout.write('\tTokenization disabled. \n')
        else:
            self.tokenizer_enabled = True
            sys.stdout.write('\tTokenization enabled.\n')
            tokenizer_encodings = Encodings(verbose=False)
            tokenizer_encodings.load(os.path.join(path_for_language, 'tokenizer.encodings'))
            config = TieredTokenizerConfig(os.path.join(path_for_language, 'tokenizer.conf'))
            tokenizer_object = TieredTokenizer(config, tokenizer_encodings, self.embeddings, runtime=True)
            tokenizer_object.load(os.path.join(path_for_language, 'tokenizer'))
            self.models[PipelineComponents.TOKENIZER] = tokenizer_object

        # 3. Load compound.
        if not os.path.isfile(os.path.join(path_for_language, 'compound.bestAcc')):
            sys.stdout.write('\tCompound disabled. \n')
        else:
            self.compound_enabled = True
            sys.stdout.write('\tCompound enabled.\n')
            compound_encodings = Encodings(verbose=False)
            compound_encodings.load(os.path.join(path_for_language, 'compound.encodings'))
            config = CompoundWordConfig(os.path.join(path_for_language, 'compound.conf'))
            compound_object = CompoundWordExpander(config, compound_encodings, self.embeddings, runtime=True)
            compound_object.load(os.path.join(path_for_language, 'compound.bestAcc'))
            self.models[PipelineComponents.COMPOUND] = compound_object

        if not os.path.isfile(os.path.join(path_for_language, 'lemmatizer.bestACC')):
            sys.stdout.write('\tLemmatizer disabled. \n')
        else:
            self.lemmatizer_enabled = True
            sys.stdout.write('\tLemmatizer enabled.\n')
            lemmatizer_encodings = Encodings(verbose=False)
            lemmatizer_encodings.load(os.path.join(path_for_language, 'lemmatizer.encodings'))
            config = LemmatizerConfig(os.path.join(path_for_language, 'lemmatizer.conf'))
            lemmatizer_object = FSTLemmatizer(config, lemmatizer_encodings, self.embeddings, runtime=True)
            lemmatizer_object.load(os.path.join(path_for_language, 'lemmatizer.bestACC'))
            self.models[PipelineComponents.LEMMATIZER] = lemmatizer_object

        if not os.path.isfile(os.path.join(path_for_language, 'tagger.bestUPOS')):
            sys.stdout.write('\tTagger disabled. \n')
        else:
            self.tagger_enabled = True
            sys.stdout.write('\tTagger enabled.\n')
            tagger_encodings = Encodings(verbose=False)
            tagger_encodings.load(os.path.join(path_for_language, 'tagger.encodings'))
            config = TaggerConfig(os.path.join(path_for_language, 'tagger.conf'))

            tagger_upos_object = BDRNNTagger(config, tagger_encodings, self.embeddings, runtime=True)
            tagger_upos_object.load(os.path.join(path_for_language, 'tagger.bestUPOS'))
            tagger_xpos_object = BDRNNTagger(config, tagger_encodings, self.embeddings, runtime=True)
            tagger_xpos_object.load(os.path.join(path_for_language, 'tagger.bestXPOS'))
            tagger_attrs_object = BDRNNTagger(config, tagger_encodings, self.embeddings, runtime=True)
            tagger_attrs_object.load(os.path.join(path_for_language, 'tagger.bestATTRS'))

            self.models[PipelineComponents.TAGGER] = [tagger_upos_object, tagger_xpos_object, tagger_attrs_object]

        if not os.path.isfile(os.path.join(path_for_language, 'parser.bestUAS')):
            sys.stdout.write('\tParser disabled. \n')
        else:
            self.parser_enabled = True
            sys.stdout.write('\tParser enabled.\n')
            lemmatizer_encodings = Encodings(verbose=False)
            lemmatizer_encodings.load(os.path.join(path_for_language, 'parser.encodings'))
            config = ParserConfig(os.path.join(path_for_language, 'parser.conf'))
            parser_object = BDRNNParser(config, lemmatizer_encodings, self.embeddings, runtime=True)
            parser_object.load(os.path.join(path_for_language, 'parser.bestUAS'))
            self.models[PipelineComponents.PARSER] = parser_object

    def get_latest_model_versions(self):
        """
        Returns a dictionary with (lang_code, latest_version) for each language code.
        """
        request = requests.get(self.MODELS_PATH_CLOUD_ALL)
        data = xmltodict.parse(request.content)

        # Make a list with all the archives in the container.
        item_names = [item['Name']
                      for item in data['EnumerationResults']['Blobs']['Blob']
                      if item['Name'].endswith('.zip')]

        # Compute latest_versions.
        latest_versions = defaultdict(str)
        for item in item_names:
            language, version = item.replace('.zip', '').split('-')
            latest_versions[language] = max(latest_versions[language], version)

        return latest_versions

    def _download_models_version(self, lang_code, version):
        """
        Downloads pre-trained models for the provided language.

        Args:
            @param lang_code: Target language code.
                See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
            @param version: Version of the models.
        """
        sys.stdout.write('Downloading models for {} \n'.format(lang_code))

        model_name = '{}-{}'.format(lang_code, version)
        model_path_cloud = os.path.join(self.cloud_path, '{}.zip'.format(model_name))
        model_path_local = os.path.join(self.disk_path, '{}.zip'.format(model_name))

        # Download and extract models for provided language.
        self._download_and_extract_lang_models(model_path_cloud, model_path_local)

        # Download Facebook embeddings.
        self._download_facebook_embeddings(lang_code)

    def _download_and_extract_lang_models(self, url, file_name, force=False):
        if os.path.exists(file_name):
            if force:
                os.remove(file_name)
            return

        # Download and extract zip archive.
        request = requests.get(url)
        request_content = request.content
        zipfile = ZipFile(io.BytesIO(request_content))
        zipfile.extractall(self.disk_path)
        zipfile.close()

    def _download_facebook_embeddings(self, lang_code):
        """
        Download Facebook embeddings for the provided lang_code.
        """
        name = self.EMBEDDINGS_NAME.format(lang_code)
        embeddings_url = self.FACEBOOK_EMBEDDINGS_URL + name
        embeddings_path = os.path.join(self.disk_path, lang_code, name)

        request = requests.get(embeddings_url)
        with open(embeddings_path, 'wb') as fd:
            fd.write(request.content)

    def version_to_donwload(self, lang_code, check_for_latest=True):
        """
        Returns the version of the language models that need to be downloaded,
        or None if there's nothing to be done.
        """
        lang_models = os.path.join(self.disk_path, lang_code)
        lang_models_version = os.path.join(lang_models, 'VERSION')

        # Get current version (if any).
        current_version = None
        if os.path.exists(lang_models):
            with open(lang_models_version) as fd:
                current_version = fd.read().strip('\n')

        # Get the latest version.
        latest_versions = self.get_latest_model_versions()
        latest_version = latest_versions.get(lang_code)

        if check_for_latest:
            if not latest_version:
                if not current_version:
                    raise ValueError('No remote version found for {}!'.format(lang_code))

                print('No remote version found for {}, using the local '
                      'version {}'.format(lang_code, current_version))
                return

            if current_version and current_version >= latest_version:
                return

            return latest_version

        if not current_version:
            return latest_version
