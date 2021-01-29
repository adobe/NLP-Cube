import sys
import optparse
import os
import tqdm


def _get_all_files(base_path):
    all_files = []
    for path, subdirs, files in os.walk(base_path):
        for name in files:
            fname = os.path.join(path, name)
            if not fname.endswith('.'):
                all_files.append(fname)
    return all_files


def _process(params):
    all_files = _get_all_files(params.wiki_base)
    f_dev = open(params.dev_file, 'w')
    f_train = open(params.train_file, 'w')
    for ii in tqdm.tqdm(range(len(all_files))):
        if (ii + 1) % params.ratio == 0:
            f = f_dev
        else:
            f = f_train
        f.write(all_files[ii] + '\n')
    f_train.close()
    f_dev.close()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--wiki', action='store', dest='wiki_base')
    parser.add_option('--train', action='store', dest='train_file')
    parser.add_option('--dev', action='store', dest='dev_file')
    parser.add_option('--ratio', action='store', default=100, type='int', dest='ratio',
                      help='train/dev ration (default=100)')

    (params, _) = parser.parse_args(sys.argv)

    if params.wiki_base and params.train_file and params.dev_file:
        _process(params)
    else:
        parser.print_help()
