import optparse
import sys

sys.path.append('')
import torch.nn as nn
import torch.nn.functional as F
from cube2.networks.text import TextEncoder
from cube2.config import TaggerConfig
from cube.io_utils.encodings import Encodings


class Tagger(nn.Module):
    encodings: Encodings
    config: TaggerConfig

    def __init__(self, config, encodings, num_languages=1):
        super(Tagger, self).__init__()
        self.config = config
        self.encodings = encodings
        self.num_languages = num_languages
        if num_languages == 1:
            lang_emb_size = None
        else:
            lang_emb_size = self.config.tagger_embeddings_size
            self.lang_emb = nn.Embedding(num_languages, lang_emb_size)
        self.text_network = TextEncoder(config, encodings, ext_conditioning=lang_emb_size)
        self.output_upos = nn.Linear(self.config.tagger_embeddings_size, len(self.encodings.upos2int))
        self.output_xpos = nn.Linear(self.config.tagger_embeddings_size, len(self.encodings.xpos2int))
        self.output_attrs = nn.Linear(self.config.tagger_embeddings_size, len(self.encodings.attrs2int))

    def forward(self, x):
        emb = self.text_network(x)
        s_upos = F.softmax(self.output_upos(emb), dim=2)
        s_xpos = F.softmax(self.output_xpos(emb), dim=2)
        s_attrs = F.softmax(self.output_attrs(emb), dim=2)
        return s_upos, s_xpos, s_attrs


def _get_tgt_labels(data, encodings):
    max_sent_size = 0
    for sent in data:
        if len(sent) > max_sent_size:
            max_sent_size = len(sent)
    tgt_upos = []
    tgt_xpos = []
    tgt_attrs = []
    for sent in data:
        upos_int = []
        xpos_int = []
        attrs_int = []
        for entry in sent:
            upos_int.append(encodings.upos2int[entry.upos])
            xpos_int.append(encodings.xpos2int[entry.xpos])
            attrs_int.append(encodings.attrs2int[entry.attrs])
        for _ in range(max_sent_size - len(sent)):
            upos_int.append(encodings.upos2int['<PAD>'])
            xpos_int.append(encodings.xpos2int['<PAD>'])
            attrs_int.append(encodings.attrs2int['<PAD>'])
        tgt_upos.append(upos_int)
        tgt_xpos.append(xpos_int)
        tgt_attrs.append(attrs_int)

    import torch
    return torch.tensor(tgt_upos), torch.tensor(tgt_xpos), torch.tensor(tgt_attrs)


def do_debug():
    from cube.io_utils.conll import Dataset
    from cube.io_utils.encodings import Encodings
    from cube2.config import TaggerConfig

    trainset = Dataset()
    devset = Dataset()
    trainset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-train.conllu', 0)
    devset.load_language('corpus/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-dev.conllu', 0)
    encodings = Encodings()
    encodings.compute(trainset, devset)
    config = TaggerConfig()
    tagger = Tagger(config, encodings, 1)

    import torch.optim as optim
    import torch.nn as nn
    trainer = optim.Adam(tagger.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for ii in range(10):

        epoch_loss = 0
        import tqdm
        for batch_idx in tqdm.tqdm(range(len(trainset.sequences) // 20)):
            data = []
            for ii in range(20):
                data.append(trainset.sequences[ii + batch_idx * 20][0])
            s_upos, s_xpos, s_attrs = tagger(data)
            tgt_upos, tgt_xpos, tgt_attrs = _get_tgt_labels(data, encodings)
            loss = criterion(s_upos.view(-1, s_upos.shape[-1]), tgt_upos.view(-1)) + criterion(
                s_xpos.view(-1, s_xpos.shape[-1]), tgt_xpos.view(-1)) + criterion(s_attrs.view(-1, s_attrs.shape[-1]),
                                                                                  tgt_attrs.view(-1))
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            epoch_loss += loss.item()
            print("\t" + str(loss.item()))

        print("epoch_loss=" + str(epoch_loss))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store_true', dest='train',
                      help='Start building a tagger model')
    parser.add_option('--debug', action='store_true', dest='debug', help='Do some standard stuff to debug the model')

    (params, _) = parser.parse_args(sys.argv)

    if params.debug:
        do_debug()
