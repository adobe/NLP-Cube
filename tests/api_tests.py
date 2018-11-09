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

class Main_Tests(unittest.TestCase):

    def setUp(self):
        # get current directory                
        self.root_path = os.path.dirname(os.path.realpath(__file__))
        self.root_path = os.path.abspath(os.path.join(self.root_path, os.pardir)) 
        self.main_file_path = os.path.join(self.root_path, "cube", "main.py")
        self.corpus_path = os.path.join(self.root_path, "tests", "test_corpus")
        self.model_path = os.path.join(self.root_path, "tests", "my_model-1.0")
        self.scratch_path = os.path.join(self.root_path, "tests", "scratch")
        self.input_file_path = os.path.join(self.corpus_path, "en_ewt-ud-test.txt")
        self.output_file_path = os.path.join(self.scratch_path, "en_ewt-ud-test-output.conllu") 
                
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.scratch_path):
            os.makedirs(self.scratch_path)
        
        print("[setUp] Absolute path of NLP-Cube: "+self.root_path)
        print()
        
    
    def test_0_init_model_store_and_list_online_models(self):
        pass
    
    def test_1_download_and_run_an_online_model(self):                                    
        command = "python3 " + self.main_file_path + " --train tokenizer"
        command+= " --train-file "+os.path.join(self.corpus_path,"en_ewt-ud-train.conllu") + " --raw-train-file " + os.path.join(self.corpus_path,"en_ewt-ud-train.txt")
        command+= " --dev-file "+os.path.join(self.corpus_path,"en_ewt-ud-dev.conllu") + " --raw-dev-file " + os.path.join(self.corpus_path,"en_ewt-ud-dev.txt")
        command+= " --embeddings "+os.path.join(self.root_path, "examples", "wiki.dummy.vec")
        command+= " --store " + os.path.join(self.model_path, "tokenizer")
        command+= " --autobatch --batch-size 1000 --set-mem 1000 --random-seed 42 --patience 1"
        print("\n\33[33m{}\n{}\33[0m".format("Tokenizer command:",command))            
        popen = subprocess.Popen(command.split(" ") , stdout=subprocess.PIPE, universal_newlines=True)
        output = []        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])
            output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        test = "Training is done with " in output[-1]
        self.assertTrue(test)
    
    
    
    def test_5_run_model(self):                                    
        command = "python3 " + self.main_file_path + " --run tokenizer,parser,tagger,lemmatizer"
        command+= " --models " + self.model_path
        command+= " --embeddings " + os.path.join(self.root_path, "examples", "wiki.dummy.vec")
        command+= " --input-file " + self.input_file_path
        command+= " --output-file " + self.output_file_path        
        print("\n\33[33m{}\n{}\33[0m".format("Model run command:",command))        
        popen = subprocess.Popen(command.split(" ") , stdout=subprocess.PIPE, universal_newlines=True)
        output = []        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])
            output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        test = "Training is done with " in output[-1]
        self.assertTrue(test)   
        
        lines = []
        with open(self.output_file_path,"r",encoding="utf8") as f:            
            line = f.readline()
            if line.strip() != "":
                lines.append(lines)
        test = "treaty" in lines[-2]
        self.assertTrue(test)
        
if __name__ == '__main__':
    unittest.main()