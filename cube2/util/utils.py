import os, sys
sys.path.insert(0, '../..')
from datetime import datetime

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
