# -*- coding: utf-8 -*-

import sys
import os
from io_utils.model_store import ModelMetadata, ModelStore
from datetime import datetime

    
if __name__ == "__main__":

    input_models_root_folder = "/work/nlpcube"
    output_models_root_folder = "/work/nlpcube_zip"    
    version = 1.0
    
    
    
    model_store = ModelStore() 
    local_models = [os.path.basename(os.path.normpath(dI)) for dI in os.listdir(input_models_root_folder) if os.path.isdir(os.path.join(input_models_root_folder,dI))]
    #print(local_models)
    
    model_tuples = [ #folder, language_code, embedding code
("UD_Afrikaans-AfriBooms","af","af"), 
("UD_Ancient_Greek-PROIEL","grc","grc"), 
("UD_Arabic-PADT","ar","ar"), 
("UD_Armenian-ArmTDP","hy","hy"), 
("UD_Basque-BDT","eu","eu"), 
("UD_Bulgarian-BTB","bg","bg"), 
("UD_Buryat-BDT","bxr","bxr"), 
("UD_Catalan-AnCora","ca","ca"), 
("UD_Chinese-GSD","zh","zh"), 
("UD_Croatian-SET","hr","hr"), 
("UD_Czech-PDT","cs","cs"), 
("UD_Danish-DDT","da","da"), 
("UD_Dutch-Alpino","nl","nl"), 
("UD_English-EWT","en","en"), 
("UD_Estonian-EDT","et","et"), 
("UD_Finnish-TDT","fi","fi"), 
("UD_French-GSD","fr","fr"), 
("UD_Galician-CTG","gl","gl"), 
("UD_German-GSD","de","de"), 
("UD_Gothic-PROIEL","got","got"), 
("UD_Greek-GDT","el","el"), 
("UD_Hebrew-HTB","he","he"), 
("UD_Hindi-HDTB","hi","hi"), 
("UD_Hungarian-Szeged","hu","hu"), 
("UD_Indonesian-GSD","id","id"), 
("UD_Irish-IDT","ga","ga"), 
("UD_Italian-ISDT","it","it"), 
("UD_Japanese-GSD","ja","ja"), 
("UD_Kazakh-KTB","kk","kk"), 
("UD_Korean-GSD","ko","ko"), 
("UD_Kurmanji-MG","kmr","ku"), 
("UD_Latin-ITTB","la","la"), 
("UD_Latvian-LVTB","lv","lv"), 
("UD_North_Sami-Giella","sme","se"), 
("UD_Norwegian-Bokmaal","no_bokmaal","no"), 
("UD_Norwegian-Nynorsk","no_nynorsk","nn"), 
("UD_Old_Church_Slavonic-PROIEL","cu","cu"), 
("UD_Persian-Seraji","fa","fa"), 
("UD_Polish-LFG","pl","pl"), 
("UD_Portuguese-Bosque","pt","pt"), 
("UD_Romanian-RRT","ro","ro"), 
("UD_Russian-SynTagRus","ru","ru"), 
("UD_Serbian-SET","sr","sr"), 
("UD_Slovak-SNK","sk","sk"), 
("UD_Slovenian-SSJ","sl","sl"), 
("UD_Spanish-AnCora","es","es"), 
("UD_Swedish-LinES","sv","sv"), 
("UD_Swedish-Talbanken","sv","sv"), 
("UD_Turkish-IMST","tr","tr"), 
("UD_Ukrainian-IU","uk","uk"), 
("UD_Upper_Sorbian-UFAL","hsb","hsb"), 
("UD_Urdu-UDTB","ur","ur"), 
("UD_Uyghur-UDT","ug","ug"), 
("UD_Vietnamese-VTB","vi","vi")]
    
    
    
    for local_model in local_models:
        print("\nPacking: "+local_model)
        
        # find correspondent
        found = False
        language_code = ""
        embedding_code = ""
        for element in model_tuples:            
            if local_model in element[0]:
                language_code = element[1]
                embedding_code = element[2]
                found = True
                break
        if not found:
            #raise Exception("Model not found in key-store!")
            print("Model not found in key-store!")
            continue
            
        metadata = ModelMetadata()
        
        metadata.language = local_model[:local_model.find("-")]                    
        # en, ro, no_nynorsk (exception)
        metadata.language_code = language_code
        # model version: 1.0, 2.1, etc. The value is a float to perform easy comparison between versions. Format must always be #.#
        metadata.model_version = version
        # *full* link to remote embeddings file 
        metadata.embeddings_remote_link = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki."+embedding_code+".vec"
        # name under which the remote file will be saved under locally
        metadata.embeddings_file_name = "wiki."+embedding_code+".vec"
        # token delimiter. Must be either space (default) or "" (for languages like Japanese, Chinese, etc.)
        if language_code in "zh ja":
            metadata.token_delimiter = "" 
        else: 
            metadata.token_delimiter = " " 
        # OPTIONAL: model build date: string
        metadata.model_build_date = str(datetime.now())
        # OPTIONAL: model build source: what corpus was it built from. Ex: UD-Romanian-RRT v2.2 
        metadata.model_build_source = local_model
        # OPTIONAL: other notes, string value
        metadata.notes = "Source: ud-treebanks-v2.2"

        metadata.info()

        input_folder = os.path.join(input_models_root_folder,local_model)
        model_store.package_model(input_folder, output_models_root_folder, metadata, should_contain_tokenizer = True, should_contain_compound_word_expander = False, should_contain_lemmatizer = True, should_contain_tagger = True, should_contain_parser = True)
        
        #break # test just one package
