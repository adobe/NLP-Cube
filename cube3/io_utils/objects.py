class Doc ():
    def __init__ (self):
        self.sentences = []

    def text(self):
        pass

    def __repr__ (self):
        pass


class Sentence():
    def __init__(self):
        self.doc = None
        self.tokens = []
        self.words = []
        self.text = None

        self._lang_id = None

    def __repr__(self):
        pass

class Token():
    def __init__(self):
        self.id = None
        self.text = None
        self.words = []
        self.other = None # ner, sentiment, etc

    def __repr__(self):
        pass

class Word():
    def __init__(self, index, word, lemma, upos, xpos, attrs, head, label, deps, space_after):
        self.index, self.is_compound_entry = self._int_try_parse(index)
        self.word = word
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.attrs = attrs
        self.head, _ = self._int_try_parse(head)
        self.label = label
        self.deps = deps
        self.space_after = space_after

    def _int_try_parse(self, value):
        try:
            return int(value), False
        except ValueError:
            return value, True

    def __repr__(self):
        return "\t".join([str(self.index), self.word if isinstance(self.word, str) else self.word.encode('utf-8'), self.lemma if isinstance(self.lemma, str) else self.lemma.encode('utf-8'), self.upos, self.xpos, self.attrs, str(self.head), self.label, self.deps, self.space_after])


