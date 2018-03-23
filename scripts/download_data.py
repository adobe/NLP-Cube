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

import sys

import optparse
import os
import logging

from utils import download_and_extract_archive, download_file

# Append parent dir to sys path.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parent_dir)

TREEBANK_CONLL = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2515/ud-treebanks-v2.1.tgz?sequence=4&isAllowed=y'
TREEBANK_LOCATION = 'corpus/ud_treebanks'

TEST_DATA_CONLL = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2184/ud-test-v2.0-conll2017.tgz?sequence=3&isAllowed=y'
TEST_DATA_LOCATION = 'corpus/ud_test'

UD_EVAL_SCRIPT = 'https://raw.githubusercontent.com/ufal/conll2017/master/evaluation_script/conll17_ud_eval.py'
UD_EVAL_LOCATION = 'cube/misc/conll17_ud_eval.py'

EMBEDDINGS_NAME = 'wiki.{}.vec'
FACEBOOK_EMBEDDINGS_URL = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/'
FACEBOOK_EMBEDDINGS_LOCATION = 'corpus/'

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--language', action='store', dest='language')
    (params, _) = parser.parse_args(sys.argv)
    if not params.language:
        print(
            '\nRun the script in the following manner:\n'
            'python scripts/download_data.py --language ro\n')
        sys.exit(1)

    # Download Treebank CONLL Universal Dependencies data.
    download_and_extract_archive(TREEBANK_CONLL, TREEBANK_LOCATION)

    # Download test CONLL Universal Dependencies data.
    download_and_extract_archive(TEST_DATA_CONLL, TEST_DATA_LOCATION)

    # Download conll17_ud_eval script
    download_file(UD_EVAL_SCRIPT, UD_EVAL_LOCATION)

    # Download Facebook embeddings for the provided language.
    name = EMBEDDINGS_NAME.format(params.language)
    language_url = FACEBOOK_EMBEDDINGS_URL + name
    location = FACEBOOK_EMBEDDINGS_LOCATION + name
    download_file(language_url, location)
