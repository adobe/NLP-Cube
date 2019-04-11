# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Append parent dir to sys path.
os.sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cube.io_utils.model_store import ModelMetadata, ModelStore
from cube.io_utils.conll import Dataset
from datetime import datetime
import time
from cube.api import Cube
 
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
    
    
if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python3 test_all_models.py path-to-UD-repository-root [OPTIONAL-path-to-local-models: where en-1.1 is, etc.]")
        print("Example: python3 test_all_models.py /work/ud-treebanks-v2.2")
        sys.exit(0)
        
    if not os.path.exists("conll18_ud_eval.py"):
        print("Please ensure that conll18_ud_eval.py is located in the same place as test_all_models.py.")
        print("ex: wget http://universaldependencies.org/conll18/conll18_ud_eval.py")
        sys.exit(0)
    from conll18_ud_eval import load_conllu_file, evaluate
    #metrics = ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]
    #example usage:     metrics_test = conll_eval(system,gold)
    #                   test_tok_f1, test_ss_f1 = metrics_test["Tokens"].f1*100., metrics_test["Sentences"].f1*100.
    def conll_eval(system_file, gold_file):    
        gold_ud = load_conllu_file(gold_file)
        system_ud = load_conllu_file(system_file)
        return evaluate(gold_ud, system_ud)
    
    if len(sys.argv)==3:
        local_model_path = sys.argv[2]
        print("Using local model path: "+local_model_path)
    else:
        local_model_path = None
        
    model_store_object = ModelStore(local_model_path) 
    local_embeddings_path = model_store_object.embeddings_repository
    local_model_path = model_store_object.disk_path
    
    online_models = model_store_object.list_online_models()
    print("Found {} online models".format(len(online_models)))
    
    local_models = model_store_object.list_local_models()
    print("Found {} local models".format(len(local_models)))
    
    
    model_count = len(online_models)
    
    # step 1. download all models
    for online_model in online_models:        
        model, version = online_model[0], online_model[1]
        if not online_model in local_models:
            print("Downloading {}-{}".format(model,version))
        else:
            print("Model {}-{} is already downloaded.".format(model,version))
            continue
        cube = Cube()
        cube.load(model, version, local_models_repository = local_model_path)
        #cube.load(model)
     
    print("\n\n")
    #for online_model in local_models: #local_models+online_models:
    for online_model in local_models+online_models:
        model, version = online_model[0], online_model[1]
        print("\n\nTesting model {}-{}, @{}".format(model,version, datetime.today()))
        if model == "pl":
            continue
        
        # go run Cube
        print("\t Reading metadata ...")        
        metadata = ModelMetadata()        
        metadata.read(os.path.join(local_model_path,model+"-"+str(version),"metadata.json"))
        
        
        mlanguage = metadata.language
        found = False        
        for elem in model_tuples:            
            if mlanguage in elem[0]:
                ud_language=elem[0]
                ud_language_code=elem[1]
                ud_embedding=elem[2]
                found=True
        if not found:
            print("Could not find model for: metadata.language="+mlanguage)
            sys.exit(0)
        
        # search for already existing processed file
        output_file = ud_language+"."+ud_language_code+"."+model+"."+str(version)+".conllu"
        if os.path.exists(output_file):
           print("\t File already processed, skipping.") 
           continue

        
        test_file_path = os.path.join(sys.argv[1],ud_language)#,ud_language_code+"-ud-test.txt")
        print("\t Searching for test file in "+test_file_path)
        all_files = os.listdir(test_file_path)
        test_file = None
        for f in all_files:
            if ".txt" in f and "-test" in f:
                test_file = f
                break
        
        if not test_file:
            print("Test file not found!")
            sys.exit(0)
        
        
        with open(os.path.join(test_file_path,test_file),"r",encoding="utf8") as f:
            content = f.readlines()
        test_data = ""
        if model == "ja" or model =="zh":
            for line in content:
                test_data+=line.replace("\r","").replace("\n","")
            test_data = test_data.replace(" ","")
        else:
            for line in content:
                test_data+=line.replace("\r","").replace("\n"," ")
            test_data = " ".join(test_data.split())
        
        print("\t Processing {} lines ...".format(len(test_data)))        
        metadata = ModelMetadata()        
        metadata.read(os.path.join(local_model_path,model+"-"+str(version),"metadata.json"))
        local_embeddings_file = os.path.join(local_embeddings_path, metadata.embeddings_file_name)
        cube = Cube(verbose=True)        
        cube.load(model, version, tokenization=True, compound_word_expanding=True, tagging=True,
             lemmatization=True, parsing=True, local_models_repository = local_model_path), 
             #local_embeddings_file = local_embeddings_file) # all enabled, including compound
        start = time.time()        
        sequences = cube(test_data)        
        end = time.time()
        
        ds = Dataset()
        ds.sequences = sequences
        ds.write(output_file)
        
        metrics = conll_eval(output_file, os.path.join(test_file_path,test_file.replace(".txt",".conllu")))
        jsmetrics = {} # apparently the metrics object is not serializable
        for metric in ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]:
            jsmetrics[metric] = {}
            jsmetrics[metric]["precision"] = 100 * metrics[metric].precision
            jsmetrics[metric]["recall"] = 100 * metrics[metric].recall
            jsmetrics[metric]["f1"] = 100 * metrics[metric].f1
            jsmetrics[metric]["aligned_accuracy"] = 100 * metrics[metric].aligned_accuracy if metrics[metric].aligned_accuracy is not None else ""
               
        #dump results
        import json
        if not os.path.exists("results.json"):
            results_object = {}
            json.dump(results_object,open("results.json","w"), indent=4, sort_keys=True)
        
        results_object = json.load(open("results.json","r"))
        
       
        elem = {}
        elem["date"] = str(datetime.today())
        elem["metrics"] = jsmetrics
        elem["test_file"] = os.path.join(test_file_path,test_file)
        elem["test_time"] = end - start
        results_object[model+"-"+str(version)] = elem
        json.dump(results_object,open("results.json","w"), indent=4, sort_keys=True)
        
    print("\n\nFinished everything.")