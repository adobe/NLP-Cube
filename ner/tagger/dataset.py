from collections import Counter


class DataRow(object):

    def __init__(self, dataset, line):
        self.orig_line = line
        self.dataset = dataset
        self._clean_up(line)

    def _clean_up(self, line):
        fields = line.strip().split("\t")
        self.word = fields[1]
        self.lemma = fields[2]
        self.UPOS = fields[3]
        self.XPOS = fields[4]
        self.attrs = fields[5]
        self.head_dep = fields[6]
        self.label_dep = fields[7]
        self.label = fields[10]


class Dataset(object):

    def __init__(self, dataset):
        train_lines = []
        lines = open(dataset, "r").readlines()
        for line in lines:
            train_lines.append(line.replace("\n", ""))

        self.sequences = self.make_sequences(train_lines)
        print dataset + " has " + str(len(self.sequences))

    def make_sequences(self, lines):
        sequences = []
        seq = ['0\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t*']
        for line in lines:
            if line != "":
                if line.startswith('#'):
                    continue
                seq.append(line)
            else:
                tmp = [DataRow(self, ll) for ll in seq]
                sequences.append(tmp)
                seq = ['0\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t<ROOT>\t*']
        return sequences


class Encodings(object):

    def __init__(self, dataset):
        self.UPOS2int = {}
        self.XPOS2int = {}
        self.attrs2int = {}
        self.word2int = {}
        self.lemma2int = {}
        self.label2int = {}
        self.label_list = []

        if dataset is not None:
            word_counter = {}
            lemma_counter = {}
            # Build the
            for sequence in dataset.sequences:
                for entry in sequence:
                    # Populate the indexes.
                    self.UPOS2int.setdefault(entry.UPOS, len(self.UPOS2int))
                    self.XPOS2int.setdefault(entry.XPOS, len(self.XPOS2int))
                    self.attrs2int.setdefault(entry.attrs, len(self.attrs2int))

                    # 1:LVC.full;2:IRFLV
                    # 1:LVC.full;2
                    if entry.label != "*":
                        labels = entry.label.split(';')
                        entry_labels = [label.split(':')[1] for label in labels if ':' in label]
                        for entry_label in entry_labels:
                            self.label2int.setdefault(entry_label, len(self.label2int))

                    # Count the number of appearences.
                    word = entry.word.decode('utf-8').lower().encode('utf-8')
                    # if word == 'prin':
                    #    print word_counter[word]
                    # if word.isalnum():
                    # print word
                    #    word = '0'

                    lemma = entry.lemma
                    # if lemma.isalnum():
                    #    lemma = '0'

                    if word not in word_counter:
                        word_counter[word] = 1
                    else:
                        word_counter[word] += 1

                    if lemma not in lemma_counter:
                        lemma_counter[lemma] = 1
                    else:
                        lemma_counter[lemma] = lemma_counter[lemma] + 1

            # Build indexes for word & lemma.
            self.word2int['<UNK>'] = 0
            self.lemma2int['<UNK>'] = 0
            for word in word_counter:
                if word_counter[word] >= 2:
                    self.word2int[word] = len(self.word2int)
            for lemma in lemma_counter:
                if lemma_counter[lemma] >= 2:
                    self.lemma2int[lemma] = len(self.lemma2int)

            self.label_list = [""] * len(self.label2int)
            for label in self.label2int:
                self.label_list[self.label2int[label]] = label

            self.input_size = len(self.word2int)

            print('Found {} UPOSes'.format(len(self.UPOS2int)))
            print('Found {} XPOSes'.format(len(self.XPOS2int)))
            print('Found {} attrs'.format(len(self.attrs2int)))
            print('Found {} labels'.format(self.label2int))
            print('Found {} label lists'.format(self.label_list))
            print('Found {} words'.format(len(self.word2int)))
            print('Found {} lemmas'.format(len(self.lemma2int)))

    def store(self, filename):
        f = open(filename, 'w')

        f.write('UPOSES ' + str(len(self.UPOS2int)) + '\n')
        for key in self.UPOS2int:
            f.write(key + '\t' + str(self.UPOS2int[key]) + '\n')
        f.write('XPOSES ' + str(len(self.XPOS2int)) + '\n')
        for key in self.XPOS2int:
            f.write(key + '\t' + str(self.XPOS2int[key]) + '\n')
        f.write('ATTRS ' + str(len(self.attrs2int)) + '\n')
        for key in self.attrs2int:
            f.write(key + '\t' + str(self.attrs2int[key]) + '\n')

        f.write('WORDS ' + str(len(self.word2int)) + '\n')
        for key in self.word2int.keys():
            f.write(key + "\t" + str(self.word2int[key]) + "\n")
        f.write('LEMMAS ' + str(len(self.lemma2int)) + '\n')
        for key in self.lemma2int:
            f.write(key + '\t' + str(self.lemma2int[key]) + '\n')

        f.write('LABELS ' + str(len(self.label2int)) + '\n')
        for key in self.label2int:
            f.write(key + '\t' + str(self.label2int[key]) + '\n')

        f.close()

    def load(self, filename):
        f = open(filename, "r")
        parts = f.readline().strip().split(' ')

        num_keys = int(parts[1])
        for i in xrange(num_keys):
            parts = f.readline().strip().split('\t')
            key = parts[0]
            value = int(parts[1])
            self.UPOS2int[key] = value

        parts = f.readline().strip().split(' ')
        num_keys = int(parts[1])
        for i in xrange(num_keys):
            parts = f.readline().strip().split('\t')
            key = parts[0]
            value = int(parts[1])
            self.XPOS2int[key] = value

        parts = f.readline().strip().split(' ')
        num_keys = int(parts[1])
        for i in xrange(num_keys):
            parts = f.readline().strip().split('\t')
            key = parts[0]
            value = int(parts[1])
            self.attrs2int[key] = value

        parts = f.readline().strip().split(' ')
        num_keys = int(parts[1])
        for i in xrange(num_keys):
            parts = f.readline().strip().split('\t')
            key = parts[0]
            value = int(parts[1])
            self.word2int[key] = value

        parts = f.readline().strip().split(' ')
        num_keys = int(parts[1])
        for i in xrange(num_keys):
            parts = f.readline().strip().split('\t')
            key = parts[0]
            value = int(parts[1])
            self.lemma2int[key] = value

        parts = f.readline().strip().split(' ')
        num_keys = int(parts[1])
        self.label_list = [""] * num_keys
        for i in xrange(num_keys):
            parts = f.readline().strip().split('\t')
            key = parts[0]
            value = int(parts[1])
            self.label2int[key] = value
            self.label_list[value] = key

        print('Loaded {} UPOSes'.format(len(self.UPOS2int)))
        print('Loaded {} XPOSes'.format(len(self.XPOS2int)))
        print('Loaded {} attrs'.format(len(self.attrs2int)))
        print('Loaded {} labels'.format(self.label2int))
        print('Loaded {} label lists'.format(self.label_list))
        print('Loaded {} words'.format(len(self.word2int)))
        print('Loaded {} lemmas'.format(len(self.lemma2int)))

        f.close()
