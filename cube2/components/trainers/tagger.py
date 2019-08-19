import sys, os
sys.path.append("../../..")

import numpy as np
import time
import torch
from tqdm import tqdm
from cube2.util.utils import parameter_count, pretty_time


def train (model, # the network model
            train_dataloader, dev_dataloader, test_dataloader, # dataloaders
            optimizer, criterion, 
            max_epochs=100000, patience=20, # training parameters
            model_store_path = None, 
            resume_training = False):        
    
    print("\nStarting training model: {}".format(model.__class__.__name__))
    
    if model_store_path is None: # saves model in the same folder as this script
        model_store_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(model_store_path): # create folder if it does not exist
        os.makedirs(model_store_path)    
    print("Working in folder {}".format(os.path.abspath(model_store_path)))    
    
    total_params, trainable_params = parameter_count(model)
    print("Model parameters: {:.2f}M, out of which trainable parameters: {:.2f}M".format(total_params/1000000, trainable_params/1000000))
        
    current_epoch = 0
    current_patience = patience
    current_epoch_time = "?"
    current_epoch_train_time = "?"
    current_epoch_dev_time = "?"
    current_epoch_test_time = "?"
    best_upos_accuracy = 0.
    best_xpos_accuracy = 0.
    best_attrs_accuracy = 0.
    
    
    if resume_training is True:        
        extra_variables = model.load(model_store_path, extension="last") # redo cand aflam cum salvam modelele
        load_optimizer_checkpoint(optimizer, model.cuda, model_store_path, extension="last") # il integram in model, nu il mai tinem separat
        if "epoch" in extra_variables:
            current_epoch = extra_variables["epoch"]                        
        print("Resuming training from epoch {}".format(current_epoch))
    
    while current_patience > 0 and current_epoch < max_epochs: 
        # INIT EPOCH ##########################################################
        print("\nStarting epoch {}: current_patience={}, time_per_epoch={} ({}/{}/{})".format(current_epoch, current_patience,  current_epoch_time, current_epoch_train_time, current_epoch_dev_time, current_epoch_test_time) )        

    
        # TRAIN ###############################################################
        time_start = time.time()
        model.train()
        total_loss = 0.
        t = tqdm(train_dataloader, ncols=120, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [train]", unit="b")
        for batch_index, batch in enumerate(t):              
            if model.cuda:
                batch = tuple([tensor.cuda() for tensor in batch])

            optimizer.zero_grad()
            
            output, loss, display_variables = model.run_batch(batch, criterion)            
                                    
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            
            optimizer.step()
             
            total_loss += loss.item()

            # update progress bar
            t_display_dict = { "loss": (total_loss / (batch_index+1)) }            
            if isinstance(display_variables, dict):
                t_display_dict.update(display_variables)                
            t.set_postfix(t_display_dict)
            
        del t
        time_train = time.time() - time_start
        
        # EVAL ################################################################
        time_start = time.time() 
        model.eval()
        total_loss = 0.
        total, upos_ok, xpos_ok, attrs_ok = 0,0,0,0
        with torch.no_grad():
            t = tqdm(dev_dataloader, ncols=120, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [valid]", unit="b")
            for batch_index, batch in enumerate(t):         
                # a batch contains (lang_id_sequences_tensor, word_sequences_tensor, word_seq_lengths, word_seq_masks, char_sequences_tensor, char_seq_lengths, char_seq_masks, symbol_sequences_tensor, symbol_seq_lengths, symbol_seq_masks, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor)                
                if model.cuda:
                    batch = tuple([tensor.cuda() for tensor in batch])
                (lang_id_sequences_tensor, word_sequences_tensor, word_seq_lengths, word_seq_masks, char_sequences_tensor, char_seq_lengths, char_seq_masks, symbol_sequences_tensor, symbol_seq_lengths, symbol_seq_masks, upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor) = batch
                    
                predicted, loss, display_variables_1 = model.run_batch(batch, criterion)
                total_loss += loss.item()
                
                # predicted is a tuple of (upos, xpos, attrs)
                # gold is (upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor)
                display_variables_2 = model.eval_batch(predicted = predicted, gold = (upos_sequences_tensor, xpos_sequences_tensor, attrs_sequences_tensor), lengths = word_seq_lengths)
                upos_ok += display_variables_2["upos_ok"]
                xpos_ok += display_variables_2["xpos_ok"]
                attrs_ok += display_variables_2["attrs_ok"]
                total += display_variables_2["total"]
                
                # update accuracies
                upos_accuracy, xpos_accuracy, attrs_accuracy = upos_ok/total, xpos_ok/total, attrs_ok/total                
                
                # update progress bar
                t_display_dict = { "loss": (total_loss / (batch_index+1)), "upos" : upos_accuracy, "xpos": xpos_accuracy, "attrs": attrs_accuracy }            
                if isinstance(display_variables, dict):
                    t_display_dict.update(display_variables_1)         
                t.set_postfix(t_display_dict)
            
            # prep for saving model at end of eval epoch
            extra = {"epoch" : current_epoch, 
                        "current_upos_accuracy": upos_accuracy, "current_xpos_accuracy": xpos_accuracy, "current_attrs_accuracy": attrs_accuracy, 
                        "best_upos_accuracy": best_xpos_accuracy, "best_xpos_accuracy": best_xpos_accuracy, "best_attrs_accuracy": best_attrs_accuracy}
                            
            if upos_accuracy > best_upos_accuracy:
                current_patience = patience
                model.save(folder = model_store_path, name = "tagger", extension = "bestUPOS", extra=extra, optimizer = optimizer, verbose=True)
            if xpos_accuracy > best_xpos_accuracy:
                current_patience = patience
                model.save(folder = model_store_path, name = "tagger", extension = "bestXPOS", extra=extra, optimizer = optimizer, verbose=True)
            if attrs_accuracy > best_attrs_accuracy:
                current_patience = patience
                model.save(folder = model_store_path, name = "tagger", extension = "bestATTRS", extra=extra, optimizer = optimizer, verbose=True)
            del t
            
        time_dev = time.time() - time_start
        
        # TEST ################################################################
        
        time_test = 0
        # to do test, same as eval
        
        # FINALIZE EPOCH ######################################################
                
        # save last model checkpoint
        #model.save_checkpoint(model_store_path, "last", extra={"epoch":current_epoch})
        #save_optimizer_checkpoint (optimizer, model_store_path, extension="last")

        current_epoch += 1
        current_patience -= 1
        current_epoch_time = pretty_time(time_train+time_dev+time_test)
        current_epoch_train_time = pretty_time(time_train)
        current_epoch_dev_time = pretty_time(time_dev)
        current_epoch_test_time = pretty_time(time_test)
        
    print("\nFinished training.")