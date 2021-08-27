import logging, re
import sys

sys.path.append('')
from cube.io_utils.objects import Document


class Encodings:
    def __init__(self, verbose=False):
        self.word_list = {}
        self.hol_word_list = []
        self.char2int = {}
        self.label2int = {}
        self.labels = []
        self.word2int = {}
        self.upos2int = {}
        self.xpos2int = {}
        self.attrs2int = {}
        self.upos_list = []
        self.xpos_list = []
        self.attrs_list = []
        self.characters = []
        self.verbose = verbose
        self.num_langs = 0

    def compute(self, train: Document, dev: Document, word_cutoff=7, char_cutoff=5, CUPT_format=False):
        if self.verbose:
            print("Computing encoding maps... ")

        self.word2int['<PAD>'] = 0
        self.hol_word_list.append('<PAD>')
        self.word2int['<UNK>'] = 1
        self.hol_word_list.append('<UNK>')
        self.char2int['<PAD>'] = 0
        self.char2int['<UNK>'] = 1
        self.char2int[' '] = 2
        self.upos2int['<PAD>'] = 0
        self.upos_list.append('<PAD>')
        self.xpos2int['<PAD>'] = 0
        self.xpos_list.append('<PAD>')
        self.attrs2int['<PAD>'] = 0
        self.attrs_list.append('<PAD>')
        self.upos2int['<UNK>'] = 1
        self.upos_list.append('<UNK>')
        self.xpos2int['<UNK>'] = 1
        self.xpos_list.append('<UNK>')
        self.attrs2int['<UNK>'] = 1
        self.attrs_list.append('<PAD>')
        self.label2int['<PAD>'] = 0
        self.labels.append('<PAD>')
        self.label2int['<UNK>'] = 1
        self.labels.append('<UNK>')

        self.characters.append("<PAD>")
        self.characters.append("<UNK>")
        self.characters.append(" ")
        char_count = {}
        word_count = {}

        for sentence in train.sentences:  # xxx
            lang_id = sentence.lang_id
            if lang_id + 1 > self.num_langs:
                self.num_langs = lang_id + 1
            for entry in sentence.words:  # entry is a Word
                word = entry.word.lower()
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] = word_count[word] + 1
                if word not in self.word_list:
                    self.word_list[word] = 0  # word is inside trainset

                uniword = entry.word.lower()
                uniword = re.sub('\d', '0', uniword)
                for i in range(len(uniword)):
                    char = uniword[i].lower()
                    if char not in char_count:
                        char_count[char] = 1
                    else:
                        char_count[char] = char_count[char] + 1

                        # if char not in self.char2int:
                        #    self.char2int[char] = len(self.char2int)

                label = entry.label

                if CUPT_format and tag_type == 'label':
                    if entry.label != "*":
                        labels = entry.label.split(';')
                        entry_labels = [label.split(':')[1] for label in labels if ':' in label]
                        for entry_label in entry_labels:
                            self.label2int.setdefault(entry_label, len(self.label2int))
                else:
                    if label not in self.label2int:
                        self.label2int[label] = len(self.label2int)
                        self.labels.append(label)

                # morphological encodings
                if entry.upos not in self.upos2int:
                    self.upos2int[entry.upos] = len(self.upos2int)
                    self.upos_list.append(entry.upos)
                if entry.xpos not in self.xpos2int:
                    self.xpos2int[entry.xpos] = len(self.xpos2int)
                    self.xpos_list.append(entry.xpos)
                if entry.attrs not in self.attrs2int:
                    self.attrs2int[entry.attrs] = len(self.attrs2int)
                    self.attrs_list.append(entry.attrs)
        if dev is not None:
            for sentence in dev.sentences:
                lang_id = sentence._lang_id
                for entry in sentence.words:
                    word = entry.word.lower()
                    if word not in self.word_list:
                        self.word_list[word] = 1  # word is inside devset only

        for word in word_count:
            if word_count[word] >= word_cutoff:
                self.word2int[word] = len(self.word2int)
                self.hol_word_list.append(word)
        for char in char_count:
            if char_count[char] >= char_cutoff and char not in self.char2int:
                self.char2int[char] = len(self.char2int)
                self.characters.append(char)

        # force add digits
        for digit in range(10):
            ds = str(digit)
            if ds not in self.char2int:
                self.char2int[ds] = len(self.char2int)
                self.characters.append(ds)
        if self.verbose:
            print("done\n")

            print("Unique words: " + str(len(self.word_list)))
            print("Unique chars: " + str(len(self.char2int)))
            print("Unique labels: " + str(len(self.label2int)))
            print("Unique UPOS: " + str(len(self.upos2int)))
            print("Unique XPOS: " + str(len(self.xpos2int)))
            print("Unique ATTRS: " + str(len(self.attrs2int)))
            print("Holistic word count: " + str(len(self.word2int)))

    def update_wordlist(self, dataset):
        for seq in dataset.sequences:
            for entry in seq:
                word = entry.word.lower()
                if word not in self.word_list:
                    self.word_list[word] = 2  # word is inside an auxiliarly set (probably test)

    def load(self, filename):
        # We only read character2int, labels, holistic words and label2int here. word_list should be recomputed for every dataset (if deemed necessary)
        with open(filename, "r", encoding="utf8") as f:
            line = f.readline()
            self.num_langs = int(line.strip().split(' ')[-1])
            line = f.readline()
            num_labels = int(line.split(" ")[1])
            if self.verbose:
                print("Loading labels " + str(num_labels))
            self.labels = [""] * num_labels
            for _ in range(num_labels):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.label2int[key] = value
                self.labels[value] = key

            line = f.readline()
            num_characters = int(line.split(" ")[1])
            self.characters = [""] * num_characters
            if self.verbose:
                print("Loading characters " + str(num_characters))
            for _ in range(num_characters):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.char2int[key] = value
                self.characters[value] = key
            line = f.readline()
            num_words = int(line.split(" ")[1])
            if self.verbose:
                print("Loading words " + str(num_words))
            for _x in range(num_words):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.word2int[key] = value

            # morphological attributes
            line = f.readline()
            num_labels = int(line.split(" ")[1])
            if self.verbose:
                print("Loading upos " + str(num_labels))
            self.upos_list = [""] * num_labels
            for _ in range(num_labels):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.upos2int[key] = value
                self.upos_list[value] = key

            line = f.readline()
            num_labels = int(line.split(" ")[1])
            self.xpos_list = [""] * num_labels
            if self.verbose:
                print("Loading xpos " + str(num_labels))
            for _ in range(num_labels):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.xpos2int[key] = value
                self.xpos_list[value] = key

            line = f.readline()
            num_labels = int(line.split(" ")[1])
            self.attrs_list = [""] * num_labels
            if self.verbose:
                print("Loading attrs " + str(num_labels))
            for _ in range(num_labels):
                line = f.readline()
                parts = line.split("\t")
                key = parts[0]
                value = int(parts[1])
                self.attrs2int[key] = value
                self.attrs_list[value] = key
            f.close()

    def save(self, filename):
        f = open(filename, "w", encoding="utf8")
        f.write("LANGS " + str(self.num_langs) + "\n")
        f.write("LABELS " + str(len(self.label2int)) + "\n")
        for label in self.label2int:
            f.write(str(label) + "\t" + str(self.label2int[label]) + "\n")
        f.write("CHARACTERS " + str(len(self.char2int)) + "\n")
        for character in self.char2int:
            f.write(character + "\t" + str(self.char2int[character]) + "\n")
        f.write("WORDS " + str(len(self.word2int)) + "\n")
        for word in self.word2int:
            f.write(word + "\t" + str(self.word2int[word]) + "\n")

        f.write("UPOS " + str(len(self.upos2int)) + "\n")
        for label in self.upos2int:
            f.write(label + "\t" + str(self.upos2int[label]) + "\n")
        f.write("XPOS " + str(len(self.xpos2int)) + "\n")
        for label in self.xpos2int:
            f.write(label + "\t" + str(self.xpos2int[label]) + "\n")
        f.write("ATTRS " + str(len(self.attrs2int)) + "\n")
        for label in self.attrs2int:
            f.write(label + "\t" + str(self.attrs2int[label]) + "\n")
        f.close()
