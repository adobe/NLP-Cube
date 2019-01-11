"""
This class should test:

1. Train a very small model with tokenizer -> parser with several options (train)
2. Test the model using the main functions (run)

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
        
        #print("\n\n"+"_"*72)
        #print("[setUp] Absolute path of NLP-Cube: "+self.root_path)
        #print()
           
    def test_1_tokenizer_training(self):                                    
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
            if stdout_line.strip()!= "":
                output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        test = "Training is done with " in output[-1]
        self.assertTrue(test)
    
    def test_2_tagger_training(self):                                    
        command = "python3 " + self.main_file_path + " --train tagger"
        command+= " --train-file "+os.path.join(self.corpus_path,"en_ewt-ud-train.conllu")
        command+= " --dev-file "+os.path.join(self.corpus_path,"en_ewt-ud-dev.conllu")
        command+= " --embeddings "+os.path.join(self.root_path, "examples", "wiki.dummy.vec")
        command+= " --store " + os.path.join(self.model_path, "tagger")
        command+= " --batch-size 500 --set-mem 1000 --patience 1"
        print("\n\33[33m{}\n{}\33[0m".format("Tagger command:",command))        
        popen = subprocess.Popen(command.split(" ") , stdout=subprocess.PIPE, universal_newlines=True)
        output = []        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])
            if stdout_line.strip()!= "":
                output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        test = "Training is done with " in output[-1]
        self.assertTrue(test)

    
        
    def test_3_lemmatizer_training(self):                                    
        command = "python3 " + self.main_file_path + " --train lemmatizer"
        command+= " --train-file "+os.path.join(self.corpus_path,"en_ewt-ud-train.conllu")
        command+= " --dev-file "+os.path.join(self.corpus_path,"en_ewt-ud-dev.conllu")
        command+= " --embeddings "+os.path.join(self.root_path, "examples", "wiki.dummy.vec")
        command+= " --store " + os.path.join(self.model_path, "lemmatizer")
        command+= " --batch-size 750 --patience 1"
        print("\n\33[33m{}\n{}\33[0m".format("Lemmatizer command:",command))        
        popen = subprocess.Popen(command.split(" ") , stdout=subprocess.PIPE, universal_newlines=True)
        output = []        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])
            if stdout_line.strip()!= "":
                output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        test = "Training is done with " in output[-1]
        self.assertTrue(test)
        
    def test_4_parser_training(self):                                    
        command = "python3 " + self.main_file_path + " --train parser"
        command+= " --train-file "+os.path.join(self.corpus_path,"en_ewt-ud-train.conllu")
        command+= " --dev-file "+os.path.join(self.corpus_path,"en_ewt-ud-dev.conllu")
        command+= " --embeddings "+os.path.join(self.root_path, "examples", "wiki.dummy.vec")
        command+= " --store " + os.path.join(self.model_path, "parser")
        command+= " --batch-size 1000 --set-mem 950 --patience 1"
        print("\n\33[33m{}\n{}\33[0m".format("Parser command:",command))        
        popen = subprocess.Popen(command.split(" ") , stdout=subprocess.PIPE, universal_newlines=True)
        output = []        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])
            if stdout_line.strip()!= "":
                output.append(stdout_line[:-1])
        popen.stdout.close()
        return_code = popen.wait()        
        test = "Training is done with " in output[-7]
        self.assertTrue(test)    

    def test_5_run_model(self):                                    
        command = "python3 " + self.main_file_path + " --run tokenizer,parser,tagger,lemmatizer"
        command+= " --models " + self.model_path
        command+= " --embeddings " + os.path.join(self.root_path, "examples", "wiki.dummy.vec")
        command+= " --input-file " + self.input_file_path
        command+= " --output-file " + self.output_file_path        
        print("\n\33[33m{}\n{}\33[0m".format("Model run command:",command))        
        popen = subprocess.Popen(command.split(" ") , stdout=subprocess.PIPE, universal_newlines=True)        
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line[:-1])        
        popen.stdout.close()
        return_code = popen.wait()                
        self.assertTrue(return_code == 0)   
        
        lines = []
        with open(self.output_file_path,"r",encoding="utf8") as f:                                                
            lines = [line for line in f.readlines() if line.strip() != ""]                            
        test = "treaty" in lines[-2]        
        self.assertTrue(test)
        
if __name__ == '__main__':
    unittest.main()