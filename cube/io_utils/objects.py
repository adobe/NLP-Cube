import sys

sys.path.append('')
from cube.io_utils.misc import fopen


class Document:
    """
    Document (document)

    A document (Document) is a collection of sentences. The structures inside are compatible with CONLL-U format. See https://universaldependencies.org/format.html for more details

    Example usage:
        doc = Document(filename='corpus/ud-treebanks-v2.7/UD_Romanian-RRT/ro_rrt-ud-dev.conllu')
    """

    def __init__(self, filename: str = None, lang_id: int = 0):
        """
        Create a new Document instance.

        You can pass optional arguments filename and lang_id if you want to load a UD-Style document
        """
        self.sentences = []
        if filename is not None:
            self.load(filename, lang_id)

    def load(self, filename: str, lang_id: int = 0):
        """
        Load a CONLL-U file into the document

        Params:
            filename - mandatory parameter
            lang_id - optional parameter (default value=0)

        Example usage:
            doc = Document()
            doc.load('corpus/ud-treebanks-v2.7/UD_Romanian-RRT/ro_rrt-ud-dev.conllu', lang_id=1000)

        """
        in_sequence = False
        f = fopen(filename, 'r')
        seq = []
        cnt = 0
        for line in f.readlines():
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            cnt += 1
            # if cnt == 100:
            #     break
            if (not line.startswith("#") or in_sequence) and line != '':
                parts = line.split("\t")

                s = Word(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7],
                         parts[8], parts[9])
                seq.append(s)
                in_sequence = True
            elif line == "":
                in_sequence = False
                if len(seq) > 0:
                    self.sentences.append(Sentence(sequence=seq, lang_id=lang_id))
                seq = []
        f.close()

    def text(self):
        self.__repr__()

    def __repr__(self):
        return '\n\n'.join([str(s) for s in self.sentences])


class Sentence:
    """
    A sentence is a collection of tokens and words. This class is not meant to be initialized outside of the Document structure
    """

    def __init__(self, sequence=None, lang_id=0, text=None):
        self.doc = None
        self.tokens = []
        self.words = []

        self.lang_id = lang_id
        skip = 0
        t = None
        if sequence is not None:
            for w in sequence:
                if '.' in str(w.index):
                    continue
                if skip == 0:
                    t = Token(index=w.index, text=w.word, space_after=(not 'spaceafter=no' in w.space_after.lower()),
                              words=[])
                    self.tokens.append(t)
                    if w.is_compound_entry:
                        parts = w.index.split('-')
                        skip = int(parts[1]) - int(parts[0]) + 1
                    else:
                        skip = 1

                if not w.is_compound_entry:
                    skip -= 1
                    w.token = t
                    t.words.append(w)
                    self.words.append(w)

        if text is None:
            self.text = self._detokenize()
        else:
            self.text = text

    def _detokenize(self):
        s = []
        for t in self.tokens:
            s.append(t.text)
            if t.space_after:
                s.append(' ')
        return ''.join(s)

    def __repr__(self):
        return '\n'.join([str(t) for t in self.tokens])


class Token:
    """
    A token contains a list of composing words. Except for Multiword Tokens, the list of words will contain a single element.
    This class is not meant to be initialized outside the Document/Sentence structures.
    """

    def __init__(self, index=0, text: str = '', words=[], space_after=True):
        self.index = index
        self.text = text
        self.words = words
        self.other = None  # ner, sentiment, etc
        self.space_after = space_after

    def __repr__(self):
        if not self.space_after:
            spa = 'SpaceAfter=No'
        else:
            spa = '_'
        head = ''
        if len(self.words) > 1:
            head = "\t".join([str(self.index), self.text,
                              '_', '_',
                              '_', '_', '_', '_', '_', spa]) + '\n'
        return head + '\n'.join([str(w) for w in self.words])


def _int_try_parse(value):
    try:
        return int(value), False
    except ValueError:
        return value, True


class Word:
    """
    Structure to hold CONLL-U style metadata for words. See https://universaldependencies.org/format.html for more details about the format
    """

    def __init__(self, index, word: str, lemma: str, upos: str, xpos: str, attrs: str, head, label: str, deps: str,
                 space_after: str, token: Token = None):
        self.index, self.is_compound_entry = _int_try_parse(index)
        self.word = word
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.attrs = attrs
        self.head, _ = _int_try_parse(head)
        self.label = label
        self.deps = deps
        self.space_after = space_after
        self.parent = token
        self.emb = None

    def __repr__(self):
        return "\t".join([str(self.index), self.word if isinstance(self.word, str) else self.word.encode('utf-8'),
                          self.lemma if isinstance(self.lemma, str) else self.lemma.encode('utf-8'), self.upos,
                          self.xpos, self.attrs, str(self.head), self.label, self.deps, self.space_after])


if __name__ == '__main__':
    print("test")
    doc = Document(filename='corpus/ud-treebanks-v2.7/UD_French-GSD/fr_gsd-ud-dev.conllu')
    print(doc)
