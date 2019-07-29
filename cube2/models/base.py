
class BaseTagger (nn.Module):
    def __init__(self, config):
        self.name = "BaseTagger"
        
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