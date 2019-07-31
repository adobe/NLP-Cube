import torch
import torch.nn as nn

class BaseTagger (nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "BaseTagger"
        
        if torch.cuda.is_available():
            print('Running on GPU.')
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            print('Running on CPU.')
            self.cuda = False
            self.device = torch.device('cpu')
        
        
    def predict(self, input):
        """
            input is a list of sentences, where each sentence is a list of ConllEntry objects
            output is the same list, with UPOS, XPOS and ATTRS tags filled in
        """
        raise Exception("BaseTagger not implemented!")
        
    def save(self, folder):
        raise Exception("BaseTagger not implemented!")
        
    def load(self, folder):
        raise Exception("BaseTagger not implemented!")