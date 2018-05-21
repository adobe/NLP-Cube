import sys

import optparse
import os
import logging
import requests

# Append parent dir to sys path.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parent_dir)

CORPUS_LANGUAGE_PATH = 'corpus/{}/'

SHARED_TASK_TRAIN = 'https://gitlab.com/parseme/sharedtask-data/raw/master/1.1/{}/train.cupt'
SHARED_TASK_TRAIN_LOCATION = 'corpus/{}/train.cupt'

SHARED_TASK_DEV = 'https://gitlab.com/parseme/sharedtask-data/raw/master/1.1/{}/dev.cupt'
SHARED_TASK_DEV_LOCATION = 'corpus/{}/dev.cupt'

logger = logging.getLogger(__name__)


def download_file(url, file_name):
    # First remove the archive_name file if it exists.
    print('Downloading file {} from {}'.format(file_name, url))
    if os.path.exists(file_name):
        os.remove(file_name)
    request = requests.get(url)
    with open(file_name, "wb") as fd:
        print('Writing content to {}'.format(file_name))
        fd.write(request.content)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--language', action='store', dest='language')
    (params, _) = parser.parse_args(sys.argv)

    language = params.language
    language = language.lower()
    if not language:
        print(
            '\nRun the script in the following manner:\n'
            'python scripts/download_data.py --language ro\n')
        sys.exit(1)

    os.makedirs(CORPUS_LANGUAGE_PATH.format(language))

    # Download shared-task scripts.
    download_file(SHARED_TASK_TRAIN.format(language.upper()), SHARED_TASK_TRAIN_LOCATION.format(language))
    download_file(SHARED_TASK_DEV.format(language.upper()), SHARED_TASK_DEV_LOCATION.format(language))
