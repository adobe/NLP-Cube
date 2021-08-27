import json
import sys
import optparse

sys.path.append('')


def _is_complete_corpus(filename):
    from cube.io_utils.conll import Dataset
    dataset = Dataset()
    dataset.load_language(filename, 0)
    ok = False
    for entry in dataset.sequences[0][0]:
        word = entry.word.replace('_', '').strip()
        if word != '':
            ok = True
            break
    return ok


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
            if _is_complete_corpus(dev_file):
                combined.append([lang_id, train_file, dev_file])
            else:
                sys.stdout.write('Removing incomplete languge: "{0}"\n'.format(lang_id))

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
                        if _is_complete_corpus(dev_file):
                            family2dataset[family].append([lang_id, train_file, dev_file])
                        else:
                            sys.stdout.write('Removing incomplete languge: "{0}"\n'.format(lang_id))
    for fam in family2dataset:
        out_file = '{1}/{0}.json'.format(fam.lower().replace(' ', '_'), params.output_file)
        json.dump(family2dataset[fam], open(out_file, 'w'), indent=4)

    # list of all languages
    all = []
    for fam in family2dataset:
        for entry in family2dataset[fam]:
            all.append(entry)

    out_file = '{0}/all.json'.format(params.output_file)
    json.dump(all, open(out_file, 'w'), indent=4)


def _process_langs(params):
    def _extract_lang_and_flavour(name):
        name = name[3:]
        parts = name.split('-')
        lang = parts[0]
        flavour = parts[1]
        return lang, flavour

    lines = open(params.input_file).readlines()
    lang2flavour = {}
    for line in lines:
        dataset_name = line.strip().split(' ')[-1]

        train_file = _get_file(params.train_base, dataset_name, 'train')
        dev_file = _get_file(params.train_base, dataset_name, 'dev')
        if dev_file == '':
            dev_file = _get_file(params.train_base, dataset_name, 'test')
        if train_file != '' and dev_file != '':
            lang_id = _get_lang_id(train_file)
            if _is_complete_corpus(dev_file):
                lang, flavour = _extract_lang_and_flavour(dataset_name)
                if lang not in lang2flavour:
                    lang2flavour[lang] = [(flavour, lang_id, train_file, dev_file)]
                else:
                    lang2flavour[lang].append((flavour, lang_id, train_file, dev_file))
    for lang in lang2flavour:
        trainset = []
        for item in lang2flavour[lang]:
            # flavour = item[0]
            lang_id = item[1]
            train_file = item[2]
            dev_file = item[3]
            trainset.append([lang_id, train_file, dev_file])
        output_file = '{0}/{1}.json'.format(params.output_file, lang.lower())
        f = open(output_file, 'w')
        json.dump(trainset, f, indent=4)
        f.close()
        sys.stdout.write('python3 cube/networks/model.py --batch-size=32 --device=cuda:0 --store=data/lemmatizer-{0} --train=examples/multilanguage/languages/{0}.json\n'.format(lang.lower()))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='input_file')
    parser.add_option('--output-file', action='store', dest='output_file')
    parser.add_option('--train-base', action='store', dest='train_base')
    parser.add_option('--type', action='store', dest='type', choices=['single', 'family', 'language'])

    (params, _) = parser.parse_args(sys.argv)

    if params.input_file and params.output_file and params.train_base:
        if params.type == 'single':
            _process_single(params)
        elif params.type == 'family':
            _process_multi(params)
        elif params.type == 'language':
            _process_langs(params)
    else:
        parser.print_help()
