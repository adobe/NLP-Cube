import json
import sys
import optparse

sys.path.append('')


def _get_list_of_folders_containing(dirName, pattern):
    import os
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath) and pattern in fullPath:
            allFiles.append(entry)
    return allFiles


def _get_list_of_files(dirName):
    import os
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        if os.path.isfile(fullPath):
            allFiles.append(fullPath)
    return allFiles


def _get_file(train_base, language, type):
    import os.path as path
    pth = path.join(train_base, language)
    files = _get_list_of_files(pth)
    for file in files:
        if file.endswith('-{0}.conllu'.format(type)):
            return file
    return ''


def _get_lang_id(file):
    parts = file.split('/')
    lang_id = parts[-1].split('-')[0]
    return lang_id


def _process_single(params):
    lines = open(params.input_file).readlines()
    combined = []
    for line in lines:
        line = line.strip()
        train_file = _get_file(params.train_base, line, 'train')
        dev_file = _get_file(params.train_base, line, 'dev')
        if dev_file == '':
            dev_file = _get_file(params.train_base, line, 'test')
        if train_file != '' and dev_file != '':
            lang_id = _get_lang_id(train_file)
            combined.append([lang_id, train_file, dev_file])

    json.dump(combined, open(params.output_file, 'w'))


def _process_multi(params):
    families = {'Germanic': 10,
                'Afro-Asiatic': 7,
                'Albanian': 1,
                'Greek': 2,
                'Armenian': 1,
                'Mande': 1,
                'Basque': 1,
                'Slavic': 13,
                'Indic': 6,
                'Celtic': 4,
                'Mongolic': 1,
                'Sino-Tibetan': 3,
                'Romance': 8,
                'Uralic': 11,
                'Austronesian': 1,
                'Japanese': 1,
                'Turkic': 3,
                'Korean': 1,
                'Iranian': 2,
                'Latin': 1,
                'Baltic': 2,
                'Tupian': 1,
                'Creole': 1,
                'Dravidian': 1,
                'Tai-Kadai': 1,
                'Austro-Asiatic': 1,
                'Pama-Nyungan': 1,
                'Niger-Congo': 1}

    family2dataset = {k: [] for k in families}
    lines = open(params.input_file).readlines()
    added_folders = {}
    for line in lines:
        line = line.strip()
        parts = line.split('  ')
        ltypes = parts[1].replace(', ', ',').split(',')
        family = ''
        for ltype in ltypes:
            if ltype in family2dataset:
                family = ltype
                break
        if family == '':
            print("Unable to process " + line)
        else:
            language_name = parts[0].split(' ')[0]
            all_folders = _get_list_of_folders_containing(params.train_base, language_name)
            if len(all_folders) == 0:
                print("Unable to find training data for " + language_name)
            else:
                for folder in all_folders:
                    if folder in added_folders:
                        continue

                    added_folders[folder] = 1
                    train_file = _get_file(params.train_base, folder, 'train')
                    dev_file = _get_file(params.train_base, folder, 'dev')
                    if dev_file == '':
                        dev_file = _get_file(params.train_base, folder, 'test')
                    if train_file != '' and dev_file != '':
                        lang_id = _get_lang_id(train_file)
                        family2dataset[family].append([lang_id, train_file, dev_file])
    for fam in family2dataset:
        out_file = '{1}/{0}.json'.format(fam.lower().replace(' ', '_'), params.output_file)
        json.dump(family2dataset[fam], open(out_file, 'w'), indent=4)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='input_file')
    parser.add_option('--output-file', action='store', dest='output_file')
    parser.add_option('--train-base', action='store', dest='train_base')
    parser.add_option('--auto', action='store_true', dest='automatic')

    (params, _) = parser.parse_args(sys.argv)

    if params.input_file and params.output_file and params.train_base:
        if not params.automatic:
            _process_single(params)
        else:
            _process_multi(params)
    else:
        parser.print_help()
