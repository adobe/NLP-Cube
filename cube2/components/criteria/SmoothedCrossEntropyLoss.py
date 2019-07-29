import torch
import torch.nn as nn

class SmoothedCrossEntropyLoss(nn.Module):
    """
    Init: 
        padding_idx
        label_smoothing: if > 0, then each int will be:
            if non padding: 1->label_smoothing , 0s-> (1-label_smoothing)/len of vocab size
            else          : 0 all the way for padding elements (where target = padding_idx)
    Input: 
        predicted: logits (no softmax or activation on them) of size [batch_size, seq_len, vocab_size]
        target: [batch_size, seq_len] with indexes
    Output:
        loss value
    
    """
    def __init__(self, ignore_index=-1, label_smoothing=1.):

        super().__init__()
        self.padding_idx = ignore_index
        self.label_smoothing = label_smoothing

        if label_smoothing < 1.:
            self.criterion = nn.KLDivLoss(reduction='batchmean') #size_average=False)            
        elif ignore_index>=0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
    def not_used_compute_loss(self, generator, dec_outs, labels):

        scores = generator(self._bottle(dec_outs)) # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            # N: the number of samples
            # M: the number of labels
            tdata = gtruth.detach()

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze() # mask of PAD
            log_likelihood = torch.gather(scores, 1, tdata.unsqueeze(1))

            one_hot = self._smooth_label(num_tokens) # Do label smoothing
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1) # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()

        loss = self.criterion(scores, gtruth)

        return loss
    
    def forward (self, predicted, target):
        # flatten tensors 
        flattened_predicted = predicted.view(-1, predicted.size(-1))
        flattened_target = target.view(-1)
        
        if self.label_smoothing < 1: # use label smoothing
            # we need to create a target distribution, on the same device, with n_classes elements for each target
            #print(flattened_target.size())
            n_classes = predicted.size(-1)
            seq_len = flattened_predicted.size(0)
            fill_value = (1.-self.label_smoothing) / (n_classes-1) # compute fill value  
            
            flattened_distribution = predicted.new_full((seq_len, n_classes), fill_value) # create new target on same device            
            flattened_distribution.scatter_(1, flattened_target.unsqueeze(1), self.label_smoothing) # puts label_smoothing values in each target distribution according to appropriate index
            mask = flattened_target.eq(self.padding_idx) # tensor([0, 0, 0, 1, 0, 1, 1, 1], dtype=torch.uint8), 1 where padding            
            flattened_distribution[mask] = 0.
            #print(flattened_distribution)            
            flattened_target = flattened_distribution.detach()
            
            # transform predicted logits to log softmax distribution
            flattened_predicted = nn.functional.log_softmax(flattened_predicted, dim = 1)            
            
        return self.criterion(flattened_predicted, flattened_target)
        
        
if __name__ == "__main__":
    import numpy as np    
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 4
    vocab_size = 3
    predicted = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float)    
    target = torch.tensor([[1,2,1,0], [1,0,0,0]], dtype=torch.long)
    #predicted = torch.tensor([[0.0,0.9,0.1],[0.0,0.5,0.5],[0.9,0.05,0.05],[0.9,0.05,0.05]], dtype=torch.float)
    #target = torch.tensor([1,2,1,0], dtype=torch.long)
    #criterion = nn.KLDivLoss(reduction='batchmean')
    #loss = criterion(predicted.log(), predicted)
    #print(loss.item())
    
    print(predicted)
    print(target)
    
    print("Loss is (smoothing = 1.): ")
    criterion = SmoothedCrossEntropyLoss(label_smoothing = 1.)    
    loss = criterion(predicted, target)
    print(loss.item())
    
    print("Loss is (smoothing = 0.9): ")
    criterion = SmoothedCrossEntropyLoss(label_smoothing = 0.9)
    loss = criterion(predicted, target)
    print(loss.item())
    