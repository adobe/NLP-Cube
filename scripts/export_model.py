# -*- coding: utf-8 -*-

import sys
import os

# Append parent dir to sys path.
os.sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cube.io_utils.model_store import ModelMetadata, ModelStore
from datetime import datetime

if __name__ == "__main__":    
    print("Usage: python3 export_model.py path-to-my-model --tokenizer(optional) --compound-word-expander(optional) --lemmatizer(optional) --tagger(optional) --parser(optional)")
    print("Example: 'python3 export_model.py path-to-my-model --tokenizer --tagger' will create a zip file named 'language_code-model_version.zip' (taken from the metadata.json) containing a tokenizer and a tagger.")
       
    # parameter checking
    _tokenizer = False
    _compound_word_expander = False  
    _lemmatizer = False  
    _parser = False  
    _tagger = False 
    model_folder_path = ""
    
    for param in sys.argv:
        if not param.startswith("--"):
            model_folder_path = param
        else:
            if "--tokenizer" in param:
                _tokenizer = True
            if "--compound-word-expander" in param:
                _compound_word_expander = True
            if "--lemmatizer" in param:
                _lemmatizer = True
            if "--tagger" in param:
                _tagger = True
            if "--parser" in param:
                _parser = True                    
    
    print("\n\tModel folder: "+model_folder_path)
    print("\tUse tokenizer: {}".format(_tokenizer))
    print("\tUse compound word expander: {}".format(_compound_word_expander))
    print("\tUse lemmatizer: {}".format(_lemmatizer))
    print("\tUse tagger: {}".format(_tagger))
    print("\tUse parser: {}\n".format(_parser))
    
    # check if path exists
    if not os.path.exists(model_folder_path):
        raise Exception ("Model folder not found!")
        
    # check if metadata exists
    if not os.path.exists(os.path.join(model_folder_path,"metadata.json")):
        raise Exception ("metadata.json not found in model folder!")
    
    # check if metadata is valid
    metadata = ModelMetadata()
    metadata.read(os.path.join(model_folder_path, "metadata.json"))
    
    output_folder_path = os.path.dirname(model_folder_path)    
    model_store_object = ModelStore(disk_path=output_folder_path)
    
    model_store_object.package_model(model_folder_path, output_folder_path, metadata, should_contain_tokenizer = _tokenizer, should_contain_compound_word_expander = _compound_word_expander, should_contain_lemmatizer = _lemmatizer, should_contain_tagger = _tagger, should_contain_parser = _parser)