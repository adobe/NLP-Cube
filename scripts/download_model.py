import optparse
import os
import sys

from utils import download_and_extract_archive, download_file

# Append parent dir to sys path.
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

MODEL_URL = 'https://s3.eu-central-1.amazonaws.com/nlp-cube/{}.tar.gz'
MODEL_LOCATION = 'corpus/trained_models/{}'

EMBEDDINGS_NAME = 'wiki.{}.vec'
FACEBOOK_EMBEDDINGS_URL = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/'
FACEBOOK_EMBEDDINGS_LOCATION = 'corpus/'


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--language', action='store', dest='language')
    (params, _) = parser.parse_args(sys.argv)
    if not params.language:
        print(
            '\nRun the script in the following manner:\n'
            'python scripts/download_model.py --language ro\n')
        sys.exit(1)

    # Download Facebook embeddings for the provided language.
    name = EMBEDDINGS_NAME.format(params.language)
    language_url = FACEBOOK_EMBEDDINGS_URL + name
    location = FACEBOOK_EMBEDDINGS_LOCATION + name
    download_file(language_url, location)

    # Download model from S3, for provided language
    model_url = MODEL_URL.format(params.language)
    model_location = MODEL_LOCATION.format(params.language)
    download_and_extract_archive(model_url, model_location)
