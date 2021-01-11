import sys, argparse

def fopen (filename, mode="r"):
    if sys.version_info[0] == 2:
        return open(filename, mode)
    else:
        if "b" in mode.lower():
            return open(filename, mode)
        else:
            return open(filename, mode, encoding="utf-8")

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Tagger ')
        self.parser.add_argument('--train', action='store', dest='train_file',
                          help='Start building a tagger model')
        self.parser.add_argument('--config', action='store', dest='config_file',
                          help='Use this configuration file for tagger')
        self.parser.add_argument('--patience', action='store', type=int, default=20, dest='patience',
                          help='Number of epochs before early stopping (default=20)')
        self.parser.add_argument('--store', action='store', dest='store', help='Output base', default='data/')
        self.parser.add_argument('--batch-size', action='store', type=int, default=16, dest='batch_size',
                          help='Number of epochs before early stopping (default=16)')
        self.parser.add_argument('--debug', action='store_true', dest='debug',
                          help='Do some standard stuff to debug the model')
        self.parser.add_argument('--resume', action='store_true', dest='resume', help='Resume training')
        self.parser.add_argument('--device', action='store', dest='device', default='cpu',
                          help='What device to use for models: cpu, cuda:0, cuda:1 ...')
        self.parser.add_argument('--test', action='store_true', dest='test', help='Test the trained model')
        self.parser.add_argument('--test-file', action='store', dest='test_file')
        self.parser.add_argument('--lang-id', action='store', dest='lang_id', type=int, default=0)
        self.parser.add_argument('--model-base', action='store', dest='model_base')

    def __call__(self, *args, **kwargs):
        return self.parser.parse_args()
