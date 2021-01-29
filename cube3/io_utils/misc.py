import sys, argparse


def fopen(filename, mode="r"):
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
        self.parser.add_argument('--patience', action='store', type=int, default=20, dest='patience',
                                 help='Number of epochs before early stopping (default=20)')
        self.parser.add_argument('--store', action='store', dest='store', help='Output base', default='data/model')
        self.parser.add_argument('--gpus', action='store', dest='gpus', type=int,
                                 help='How many GPUs to use (default=1)', default=1)
        self.parser.add_argument('--num-workers', action='store', dest='num_workers', type=int,
                                 help='How many dataloaders to use (default=4)', default=4)
        self.parser.add_argument('--batch-size', action='store', type=int, default=16, dest='batch_size',
                                 help='Batch size (default=16)')
        self.parser.add_argument('--debug', action='store_true', dest='debug',
                                 help='Do some standard stuff to debug the model')
        self.parser.add_argument('--resume', action='store_true', dest='resume', help='Resume training')
        self.parser.add_argument('--lm-model', action='store', dest='lm_model', default='transformer:xlm-roberta-base',
                                 help='What LM model to use (default=transformer:xlm-roberta-base)')
        self.parser.add_argument('--lm-device', action='store', dest='lm_device', default='cuda:0',
                                 help='Where to load LM (default=cuda:0)')
        self.parser.add_argument('--config', action='store', dest='config_file', help='Load config file')

    def __call__(self, *args, **kwargs):
        return self.parser.parse_args()
