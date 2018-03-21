import sys

import optparse
import os

# Append parent dir to sys path.
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--language', action='store', dest='language')
    (params, _) = parser.parse_args(sys.argv)
    if not params.language:
        print(
            '\nRun the script in the following manner:\n'
            'python scripts/download_model.py --language ro\n')
        sys.exit(1)


    # TODO Download model
    # TODO Download Facebook embeddings for the provided language
