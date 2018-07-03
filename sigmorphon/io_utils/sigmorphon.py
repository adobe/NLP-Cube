class Sigmorphon2CONLL:
    def __init__(self):
        self.sequences = []

    def read_from_file(self, filename):
        words_in_seq = 50
        seq = []
        with open(filename, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "").replace("\r", "")
                parts = line.split("\t")
                src = parts[0]
                dst = parts[1]
                morph = parts[2]

                from conll import ConllEntry
                entry = ConllEntry(len(seq) + 1, src, dst, morph, "_", "_", 0, "_", "", "_")
                seq.append(entry)

                if len(seq) == words_in_seq:
                    self.sequences.append(seq)
                    seq = []
            if len(seq) != 0:
                self.sequences.append(seq)
            f.close()

    def convert2conll(self):
        from conll import Dataset
        ds = Dataset()
        ds.sequences = self.sequences
        return ds
