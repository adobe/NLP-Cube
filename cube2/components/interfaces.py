import os
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
        
    def run_batch(self, *kargs, **kwargs):
        raise Exception("BaseTagger not implemented!")
        
    def predict(self, input):
        """
            input is a list of sentences, where each sentence is a list of ConllEntry objects
            output is the same list, with UPOS, XPOS and ATTRS tags filled in
        """
        raise Exception("BaseTagger not implemented!")
        
    def save(self, folder, name, extension, extra={}, optimizer=None, verbose=False):
        """
            extra is a dict that contains key-value pairs that will be saved with the model
        """
        filename = os.path.join(folder, name + "." + extension)
        if verbose:
            print("Saving model to [{}] ... ".format(filename))
        checkpoint = {}
        checkpoint["state_dict"] = self.state_dict()
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["extra"] = extra
        torch.save(checkpoint, filename)

    def load(self, folder, name, extension, optimizer=None, verbose=False):        
        """
            pass optimizer object to load the optimizer's parameters as well
        """
        filename = os.path.join(folder, name + "." + extension)
        if verbose:
            print("Loading model from [{}] ...".format(filename))        
        checkpoint = torch.load(filename)
        
        self.load_state_dict(checkpoint["state_dict"])        
        self.to(self.device)
    
        if optimizer is not None: # we also load the optimizer
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.cuda:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
                        
        return checkpoint["extra"]