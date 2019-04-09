"""
This class should test:

0. Init the model_store object and list online models
1. Download and run an online model
2. Run a local model (should be created with main_tests.py before in tests/my_model-1.0)
3.1. Package a local model *without* embeddings link set in metadata.json
3.2. Import it into NLP-Cube
3.3. Run the local model with manual embeddings

4.1. Package a local model *with* embeddings link set in metadata.json
4.2. Import it into NLP-Cube
4.3. Run the local model without manual embeddings

"""
import os, sys, subprocess
import unittest

class Api_Tests(unittest.TestCase):

    def setUp(self):
        # get current directory                
        self.root_path = os.path.dirname(os.path.realpath(__file__))
        self.root_path = os.path.abspath(os.path.join(self.root_path, os.pardir)) 
        self.main_file_path = os.path.join(self.root_path, "cube", "main.py")
        self.scripts_path = os.path.join(self.root_path, "scripts")
        self.corpus_path = os.path.join(self.root_path, "tests", "test_corpus")
        self.model_path = os.path.join(self.root_path, "tests", "my_model-1.0")
        self.local_model_repo = os.path.join(self.root_path, "tests")
        self.scratch_path = os.path.join(self.root_path, "tests", "scratch")
        self.input_file_path = os.path.join(self.corpus_path, "en_ewt-ud-test.txt")
        self.output_file_path = os.path.join(self.scratch_path, "en_ewt-ud-test-output.conllu") 
                
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.scratch_path):
            os.makedirs(self.scratch_path)
                
        #import root_path    
        sys.path.append(self.root_path)
        
        #print("[setUp] Absolute path of NLP-Cube: "+self.root_path)
        #print()
        
    
    def test_0_init_model_store_and_list_online_models(self):       
        print("\n\33[33m{}\33[0m".format("0. Loading the model store and querying the online database ..."))
        from cube.io_utils.model_store import ModelMetadata, ModelStore
        model_store_object = ModelStore()
        online_models = model_store_object.list_online_models()
        print("Found "+str(len(online_models))+ " models online.")      
        self.assertTrue(len(online_models)>0)
    
    def test_1_1_download_and_run_an_online_model_latest_version(self):                                    
        print("\n\33[33m{}\33[0m".format("1.1 Loading an online model (latest_version) ..."))
        from cube.api import Cube
        cube = Cube(verbose=True)
        #cube.load('en_small', tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True)
        cube.load('bxr', tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True)
        cube.metadata.info()
        text = "I'm a success today because I had a friend who believed in me and I didn't have the heart to let him down. This is a quote by Abraham Lincoln."
        sentences = cube(text)
        self.assertTrue(len(sentences)>0)
        self.assertTrue(len(sentences[0])>0)        
    
    def test_1_2_download_and_run_an_online_model_specific_version(self):                                    
        print("\n\33[33m{}\33[0m".format("1.2. Loading an online model (sme, 1.0) ..."))
        from cube.api import Cube
        cube = Cube(verbose=True)
        cube.load('sme', version='1.0', tokenization=True, compound_word_expanding=False, tagging=False, lemmatization=False, parsing=False)
        cube.metadata.info()
        text = "I'm a success today because I had a friend who believed in me and I didn't have the heart to let him down. This is a quote by Abraham Lincoln."
        sentences = cube(text)
        self.assertTrue(len(sentences)>0)
        self.assertTrue(len(sentences[0])>0)        
    
    # This test needs my_model-1.0 to be locally created with main_tests.py
    def test_2_run_a_local_model(self):  
        print("\n\33[33m{}\33[0m".format("2. Run a local model that does not have embeddings or metadata (running with dummy.vec embeddings) ..."))
        embeddings = os.path.join(self.root_path, "examples","wiki.dummy.vec")
        from cube.api import Cube
        cube = Cube(verbose=True)
        cube.load('my_model', tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True, local_models_repository=self.local_model_repo, local_embeddings_file=embeddings)        
        text = "I'm a success today because I had a friend who believed in me and I didn't have the heart to let him down. This is a quote by Abraham Lincoln."
        sentences = cube(text)
        self.assertTrue(len(sentences)>0)
        self.assertTrue(len(sentences[0])>0)   
        
    
    def test_3_1_package_a_local_model_without_embeddings_link_in_metadata(self):  
        print("\n\33[33m{}\33[0m".format("3.1. Package a local model without an embeddings file ..."))
        
        # create metadata file
        with open(os.path.join(self.model_path,"metadata.json"),"w",encoding="utf-8") as f:
            f.write("{\n")            
            f.write('"embeddings_file_name": "wiki.dummy.vec",\n')
            f.write('"embeddings_remote_link": "",\n')
            f.write('"language": "UD_English",\n')
            f.write('"language_code": "my_model",\n')
            f.write('"model_build_date": "2020-01-01",\n')
            f.write('"model_build_source": "UD_English-ParTuT",\n')
            f.write('"model_version": 1.0,\n')
            f.write('"notes": "Source: ud-treebanks-v2.2, dummy model",\n')
            f.write('"token_delimiter": " "\n')
            f.write("}\n")            
        
        #python3 /work/NLP-Cube/scripts/export_model.py /work/my_model-1.0 --tokenizer --tagger
        command = "python3 " + os.path.join(self.scripts_path, "export_model.py") + " " + self.model_path
        command+= " --tokenizer --tagger --parser --lemmatizer"
        print("\n\t\t\33[32m{}\n{}\33[0m".format("Export command:",command))        
        ''' popen = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE, universal_newlines=True)        
        output = []        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])
            if stdout_line.strip()!= "":
                output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        '''
        os.system(command)
        
        test = os.path.exists(os.path.join(self.local_model_repo,"my_model-1.0.zip"))
        self.assertTrue(test)
    
    def test_3_2_import_model_in_store(self):      
        print("\n\33[33m{}\33[0m".format("3.2. Import locally created model in store ..."))        
        command = "python3 " + os.path.join(self.scripts_path, "import_model.py") + " " + os.path.join(self.local_model_repo,"my_model-1.0.zip")        
        print("\n\t\t\33[32m{}\n{}\33[0m".format("Import command:",command))        
        '''popen = subprocess.Popen(command.split(" ") , stdout=subprocess.PIPE, universal_newlines=True)
        output = []        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])
            if stdout_line.strip()!= "":
                output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        '''
        os.system(command)
         
        # check it is in store
        from cube.io_utils.model_store import ModelMetadata, ModelStore
        model_store_object = ModelStore()
        local_models = model_store_object.list_local_models()
        test = False
        for model, version in local_models:
            if model == "my_model":  
                test = True
        self.assertTrue(test)
        
    
    def test_3_3_run_model_with_manual_embeddings(self):  
        print("\n\33[33m{}\33[0m".format("3.3. Run a local model with manual embeddings ..."))                
        embeddings = os.path.join(self.root_path, "examples","wiki.dummy.vec")
        print("\t\tPath to local manual embeddings file: "+embeddings)
        from cube.api import Cube
        cube = Cube(verbose=True)
        cube.load('my_model', tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True, local_embeddings_file=embeddings)        
        text = "I'm a success today because I had a friend who believed in me and I didn't have the heart to let him down. This is a quote by Abraham Lincoln."
        sentences = cube(text)
        self.assertTrue(len(sentences)>0)
        self.assertTrue(len(sentences[0])>0)   

    def test_4_1_package_a_local_model_with_embeddings_link_in_metadata(self):  
        print("\n\33[33m{}\33[0m".format("4.1. Package a local model with an external embeddings file link..."))
        
        # create metadata file
        with open(os.path.join(self.model_path,"metadata.json"),"w",encoding="utf-8") as f:
            f.write("{\n")            
            f.write('"embeddings_file_name": "wiki.got.vec",\n')
            f.write('"embeddings_remote_link": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.got.vec",\n')            
            f.write('"language": "UD_English",\n')
            f.write('"language_code": "my_model",\n')
            f.write('"model_build_date": "2020-01-01",\n')
            f.write('"model_build_source": "UD_English-ParTuT",\n')
            f.write('"model_version": 1.0,\n')
            f.write('"notes": "Source: ud-treebanks-v2.2, dummy model, -got- embeddings because they are small",\n')
            f.write('"token_delimiter": " "\n')
            f.write("}\n")            
        
        # first cleanup if my_model-1.0.zip already exists
        if os.path.exists(os.path.join(self.local_model_repo,"my_model-1.0.zip")):
            os.remove(os.path.join(self.local_model_repo,"my_model-1.0.zip"))
        
        command = "python3 " + os.path.join(self.scripts_path, "export_model.py") + " " + self.model_path
        command+= " --tokenizer --tagger --parser --lemmatizer"
        print("\n\t\t\33[32m{}\n{}\33[0m".format("Export command:",command))        
        '''popen = subprocess.Popen(command.split(" ") , stdout=subprocess.PIPE, universal_newlines=True)
        output = []        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])
            if stdout_line.strip()!= "":
                output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        '''
        os.system(command)
        test = os.path.exists(os.path.join(self.local_model_repo,"my_model-1.0.zip"))
        self.assertTrue(test)
    
    def test_4_2_import_model_in_store(self):  
        print("\n\33[33m{}\33[0m".format("4.2. Import locally created model in store (with prior cleanup)..."))        
        
        # first check local models         
        from cube.io_utils.model_store import ModelMetadata, ModelStore
        model_store_object = ModelStore()
        local_models = model_store_object.list_local_models()
        print("\tFound local models:"+str(local_models))        
        self.assertTrue(len(local_models)>0)
        
        # search for my_model
        for model, version in local_models:
            if model == "my_model":                                
                # delete local model
                print("\tDeleting 'my_model-1.0'...")        
                model_store_object.delete_model("my_model","1.0")                
                local_models_new = model_store_object.list_local_models()
                print("\tFound local models:"+str(local_models_new))        
                self.assertTrue(len(local_models)>len(local_models_new))
                
        # import new model
        command = "python3 " + os.path.join(self.scripts_path, "import_model.py") + " " + os.path.join(self.local_model_repo,"my_model-1.0.zip")        
        print("\n\t\t\33[32m{}\n{}\33[0m".format("Import command:",command))        
        '''popen = subprocess.Popen(command.split(" ") , stdout=subprocess.PIPE, universal_newlines=True)
        output = []        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])
            if stdout_line.strip()!= "":
                output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        '''
        os.system(command)
        
        test = os.path.exists(os.path.join(self.local_model_repo,"my_model-1.0.zip"))
        self.assertTrue(test)
        
        # check it is in store
        local_models = model_store_object.list_local_models()
        test = False
        for model, version in local_models:
            if model == "my_model":  
                test = True
        self.assertTrue(test)
        
    def test_4_3_run_model_with_default_external_embeddings(self):  
        print("\n\33[33m{}\33[0m".format("4.3. Run a local model with default external embeddings ..."))                        
        from cube.api import Cube
        cube = Cube(verbose=True)
        cube.load('my_model', tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True)        
        text = "I'm a success today because I had a friend who believed in me and I didn't have the heart to let him down. This is a quote by Abraham Lincoln."
        sentences = cube(text)
        self.assertTrue(len(sentences)>0)
        self.assertTrue(len(sentences[0])>0)       
    
    def test_5_cleanup(self):
        print("\n\33[33m{}\33[0m".format("5. Cleanup after myself ..."))                        
        
        # delete my_model from the store, if it exists
        from cube.io_utils.model_store import ModelMetadata, ModelStore
        model_store_object = ModelStore()
        local_models = model_store_object.list_local_models()
        print("\tFound local models:"+str(local_models))        
        self.assertTrue(len(local_models)>0)
        
        for model, version in local_models:
            if model == "my_model":                                
                # delete local model
                print("\tDeleting 'my_model-1.0'...")        
                model_store_object.delete_model("my_model","1.0")                
                local_models_new = model_store_object.list_local_models()
                print("\tFound local models:"+str(local_models_new))        
                self.assertTrue(len(local_models)>len(local_models_new))
                break
    
        # delete my_model.zip, if it exists
        if os.path.exists(os.path.join(self.local_model_repo,"my_model-1.0.zip")):
            os.remove(os.path.join(self.local_model_repo,"my_model-1.0.zip"))        
        self.assertFalse(os.path.exists(os.path.join(self.local_model_repo,"my_model-1.0.zip")))
        
    
if __name__ == '__main__':        
    unittest.main()