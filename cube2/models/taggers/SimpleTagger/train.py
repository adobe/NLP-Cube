import sys, subprocess, os
sys.path.append("../../../..")

import numpy as np
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from cube2.models.taggers.SimpleTagger.network import SimpleTagger
from cube2.components.lookup import Lookup, createLookup
from cube2.components.loaders.loaders import getSequenceDataLoader

def get_freer_gpu():
    try:    
        import numpy as np
        os_string = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True).decode("utf-8").strip().split("\n")    
        memory_available = [int(x.strip().split()[2]) for x in os_string]
        return int(np.argmax(memory_available))
    except:
        print("Warning: Could execute 'nvidia-smi', default GPU selection is id=0")
        return 0

def train (network, train_dataloader, dev_dataloader, test_dataloader, optimizer, criterion):    
    current_epoch = 0
    network.train()
    while True:
        total_loss, log_average_loss = 0, 0        
        t = tqdm(train_dataloader, ncols=120, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [train]", unit="b")
        for batch_index, batch in enumerate(t):  
            if network.cuda:
                (lang_id_sequences_tensor, seq_lengths, word_sequences_tensor, char_sequences_tensor, symbol_sequences_tensor, seq_masks, char_seq_lengths, symbol_seq_lengths, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor) = batch 
                lang_id_sequences_tensor = lang_id_sequences_tensor.cuda()
                seq_lengths = seq_lengths.cuda()
                word_sequences_tensor = word_sequences_tensor.cuda()
                char_sequences_tensor = char_sequences_tensor.cuda()
                symbol_sequences_tensor = symbol_sequences_tensor.cuda()
                seq_masks = seq_masks.cuda()
                char_seq_lengths = char_seq_lengths.cuda()
                symbol_seq_lengths = symbol_seq_lengths.cuda()
                upos_sequences_tensor = upos_sequences_tensor.cuda()
                xpos_sequences_tensor = xpos_sequences_tensor.cuda()
                attrs_sequences_tensor = attrs_sequences_tensor.cuda()
                batch = (lang_id_sequences_tensor, seq_lengths, word_sequences_tensor, char_sequences_tensor, symbol_sequences_tensor, seq_masks, char_seq_lengths, symbol_seq_lengths, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor)
            
            optimizer.zero_grad()
            
            s_upos, s_xpos, s_attrs = network.forward(batch)
            
            #print(s_upos.size())
            #print(s_upos.view(-1,len(network.lookup.upos2int)).size())
            #print(upos_sequences_tensor.view(-1).size())
            loss = (criterion(s_upos.view(-1,len(network.lookup.upos2int)), upos_sequences_tensor.view(-1)) +
                    criterion(s_xpos.view(-1,len(network.lookup.xpos2int)), xpos_sequences_tensor.view(-1)) +
                    criterion(s_attrs.view(-1,len(network.lookup.attrs2int)), attrs_sequences_tensor.view(-1)))
            
            #loss = (criterion(s_upos.view(-1, s_upos.shape[-1]), tgt_upos.view(-1)) +
                    #criterion(s_xpos.view(-1, s_xpos.shape[-1]), tgt_xpos.view(-1)) +
                    #criterion(s_attrs.view(-1, s_attrs.shape[-1]), tgt_attrs.view(-1)))
            
            
            #loss = criterion(output.view(-1, n_class), y_batch.contiguous().flatten())
            
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.) # parametrize clip value TODO, also what is a good value? 0.1, 0.5, 1 or 5?            
            
            optimizer.step()
             
            total_loss += loss.item()
            log_average_loss = total_loss / (batch_index+1)
            
            t.set_postfix(loss=log_average_loss, cur_loss = loss.item())
            
        del t
        
        network.eval()
        with torch.no_grad():
            total_loss, log_average_loss = 0, 0  
            total, upos_ok, xpos_ok, attrs_ok = 0,0,0,0            
            t = tqdm(dev_dataloader, ncols=120, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [valid]", unit="b")
            for batch_index, batch in enumerate(t):         
                if network.cuda:
                    (lang_id_sequences_tensor, seq_lengths, word_sequences_tensor, char_sequences_tensor, symbol_sequences_tensor, seq_masks, char_seq_lengths, symbol_seq_lengths, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor) = batch 
                    lang_id_sequences_tensor = lang_id_sequences_tensor.cuda()
                    seq_lengths = seq_lengths.cuda()
                    word_sequences_tensor = word_sequences_tensor.cuda()
                    char_sequences_tensor = char_sequences_tensor.cuda()
                    symbol_sequences_tensor = symbol_sequences_tensor.cuda()
                    seq_masks = seq_masks.cuda()
                    char_seq_lengths = char_seq_lengths.cuda()
                    symbol_seq_lengths = symbol_seq_lengths.cuda()
                    upos_sequences_tensor = upos_sequences_tensor.cuda()
                    xpos_sequences_tensor = xpos_sequences_tensor.cuda()
                    attrs_sequences_tensor = attrs_sequences_tensor.cuda()
                    batch = (lang_id_sequences_tensor, seq_lengths, word_sequences_tensor, char_sequences_tensor, symbol_sequences_tensor, seq_masks, char_seq_lengths, symbol_seq_lengths, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor)   
                
                s_upos, s_xpos, s_attrs = network.forward(batch)
                
                for b_idx in range(len(batch)):
                    for w_idx in range(seq_lengths[b_idx]):
                        total += 1                        
                        pred_upos = np.argmax(s_upos[b_idx, w_idx].detach().cpu())
                        pred_xpos = np.argmax(s_xpos[b_idx, w_idx].detach().cpu())
                        pred_attrs = np.argmax(s_attrs[b_idx, w_idx].detach().cpu())
                    
                        if pred_upos == upos_sequences_tensor[b_idx, w_idx].detach().cpu():
                            upos_ok += 1
                        if pred_xpos == xpos_sequences_tensor[b_idx, w_idx].detach().cpu():
                            xpos_ok += 1
                        if pred_attrs == attrs_sequences_tensor[b_idx, w_idx].detach().cpu():
                            attrs_ok += 1

            
            print("UPOS {}, XPOS {}, ATTRS {}".format(upos_ok / total, xpos_ok / total, attrs_ok / total))
            print()
        current_epoch += 1



# GPU SELECTION ########################################################
if torch.cuda.is_available():
    freer_gpu = get_freer_gpu()
    print("Auto-selected GPU {} id {}".format(torch.cuda.get_device_name(freer_gpu),freer_gpu))
    torch.cuda.set_device(freer_gpu)
# ######################################################################


print("\n\n\n")
#lookup = createLookup(["d:\\ud-treebanks-v2.4\\UD_Romanian-RRT\\ro_rrt-ud-train.conllu"], verbose=True)      
#lookup.save("../../../../scratch")
lookup = Lookup("../../../../scratch")


train_dataloader = getSequenceDataLoader(["../../../../../ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-train.conllu"], batch_size=32, lookup_object=lookup, num_workers=0, shuffle=True)
dev_dataloader = getSequenceDataLoader(["../../../../../ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-dev.conllu"], batch_size=256, lookup_object=lookup, num_workers=0, shuffle=False)
#test_dataloader = getSequenceDataLoader(["d:\\ud-treebanks-v2.4\\UD_Romanian-RRT\\ro_rrt-ud-test.conllu"], batch_size=2, lookup_object=lookup, num_workers=0, shuffle=True)
network = SimpleTagger(lookup)

#print(network)

optimizer = torch.optim.SGD(network.parameters(), lr=.1, momentum=0.9)
criterion = nn.CrossEntropyLoss(ignore_index=0)

train(network, train_dataloader, dev_dataloader, None, optimizer, criterion)
