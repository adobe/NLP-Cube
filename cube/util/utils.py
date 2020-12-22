import os, sys
sys.path.insert(0, '../..')
from datetime import datetime

def use_gpu(gpu_id = None, verbose = False):
    def _get_freer_gpu():
        try:    
            import numpy as np
            import subprocess
            os_string = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True).decode("utf-8").strip().split("\n")    
            memory_available = [int(x.strip().split()[2]) for x in os_string]
            return int(np.argmax(memory_available))
        except:
            print("Warning: Could not execute 'nvidia-smi' on this platform, selecting default GPU id = 0")
            return 0

    import torch
    if torch.cuda.is_available():
        if gpu_id is None: # auto select GPU
            freer_gpu = _get_freer_gpu()
            if verbose:
                print("Auto-selecting CUDA device #{}: {}".format(freer_gpu, torch.cuda.get_device_name(freer_gpu)))
            torch.cuda.set_device(freer_gpu)
        else:
            if not isinstance(gpu_id, int):
                raise Exception("ERROR: Please specify the GPU id as a valid integer!")
            if verbose:
                print("Selected CUDA device #{}: {}".format(gpu_id, torch.cuda.get_device_name(gpu_id)))
            torch.cuda.set_device(gpu_id)

def parameter_count (model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
        
def pretty_time(seconds, granularity=2):
    intervals = (('w', 604800),('d', 86400),('h', 3600),('m', 60),('s', 1))
    result = []    
    seconds = int(seconds)    
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count            
            result.append("{}{}".format(value, name))
        return ':'.join(result[:granularity])
        
def pretty_sequences(a, b):
    CRED = '\033[91m'
    CEND = '\033[0m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    max_len = 0
    for i in range(len(a)):
        if len(str(a[i])) > max_len:
            max_len = len(str(a[i]))
    for i in range(len(b)):
        if len(str(b[i])) > max_len:
            max_len = len(str(b[i]))
    str_a = ""
    str_b = ""
    for i in range(min(len(a), len(b))):
        str_a += str(a[i]).rjust(max_len)+" "
        if a[i] == b[i]:            
            str_b += CGREEN+str(b[i]).rjust(max_len)+CEND+" "
        else:            
            str_b += CRED+str(b[i]).rjust(max_len)+CEND+" "
    if len(a)>len(b):
        for i in range(len(a)):
            str_a += str(a[i]).rjust(max_len)+" "
    elif len(a)<len(b):
        for i in range(len(b)):
            str_b += CYELLOW+str(b[i]).rjust(max_len)+CEND+" "
    
    print(str_a)
    print(str_b)
