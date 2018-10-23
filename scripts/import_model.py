# -*- coding: utf-8 -*-

import sys
import os

# Append parent dir to sys path.
os.sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cube.io_utils.model_store import ModelMetadata, ModelStore
from datetime import datetime

if __name__ == "__main__":        
    if len(sys.argv)!=2:
        print("Usage: python3 import_model.py path-to-my-model")
        print("Example: 'python3 import_model.py my_model-1.0.zip")
        sys.exit(0)
        
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        raise Exception("File {} not found!".format(model_path))
       
    if ".zip" not in model_path:
        raise Exception("File {} must be a zip! See export_model.py and the online tutorial on how to package a locally trained model!".format(model_path))
    
    model_store_object = ModelStore()
    
    model_store_object.import_local_model(model_path)
   