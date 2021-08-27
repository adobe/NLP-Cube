# -*- coding: utf-8 -*-

from collections import defaultdict
import io
import os
import sys
import json
import tempfile
from shutil import rmtree, copyfile
from cube.misc.misc import fopen
import zipfile
from zipfile import ZipFile
from pathlib import Path

import requests
import xmltodict

# Append parent dir to sys path.
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)
   
class ModelMetadata(object):
    def __init__(self):
        # language of model: English, Spanish
        self.language = ""        
        # language_code: en, es, de, got, no_nynorsk, etc. (note: no "-" allowed in the language code!)
        self.language_code = ""
        # model version: 1.0, 2.1, etc. The value is a float to perform easy comparison between versions. Format must always be #.#
        self.model_version = 1.0
        # *full* link to remote embeddings file 
        self.embeddings_remote_link = ""
        # name under which the remote file will be saved under locally
        self.embeddings_file_name = ""
        # token delimiter. Must be either space (default) or "" (for languages like Japanese, Chinese, etc.)
        self.token_delimiter = " " 
        # minimum NLP Cube version required to run this model (string, in the xx.xx.xx format)
        self.minimum_nlp_cube_version = ""                
        # OPTIONAL: model build date: string
        self.model_build_date = ""
        # OPTIONAL: model build source: what corpus was it built from. Ex: UD-Romanian-RRT v2.2 
        self.model_build_source = ""
        # OPTIONAL: other notes, string value
        self.notes = ""
        
    
    def read(self, filename):                            
        if not os.path.exists(filename):
            raise Exception("Metadata file ["+filename+"] not found!")
        if not filename.endswith("metadata.json"):
            raise Exception("Metadata file ["+filename+"] does not seem to be valid!")
        with fopen(filename,"r") as f:
            data = json.load(f)
        if sys.version_info[0] == 2: 
            items = data.iteritems()
        else:
            items = data.items()
        for key, value in items:
            if key == "model_version": # safety check to keep the version as float
                if isinstance(value,str):
                    self.__dict__[key] = float(value)
            else:
                self.__dict__[key] = value
    
    def save(self, filename):
        assert(filename.endswith("metadata.json"))
        obj = {}
        obj["language"] = self.language        
        obj["language_code"] = self.language_code
        obj["model_version"] = self.model_version
        obj["minimum_nlp_cube_version"] = self.minimum_nlp_cube_version
        obj["embeddings_remote_link"] = self.embeddings_remote_link
        obj["embeddings_file_name"] = self.embeddings_file_name
        obj["token_delimiter"] = self.token_delimiter
        obj["model_build_date"] = self.model_build_date
        obj["model_build_source"] = self.model_build_source
        obj["notes"] = self.notes
        with fopen(filename,"w") as f:
            json.dump(obj, f, indent=4, sort_keys=True)    
    
    def check_nlp_cube_compatibility(self, nlp_cube_version): 
        if self.minimum_nlp_cube_version == None:
            return True 
        if self.minimum_nlp_cube_version == "":
            return True    
        
        cube_parts = str(nlp_cube_version).strip().split(".")
        model_parts = str(self.minimum_nlp_cube_version).strip().split(".")      
        # for cube
        cube_version = 0        
        cube_version += int(cube_parts[0].rjust(2,'0').ljust(8,'0'))
        cube_version += int(cube_parts[1].rjust(2,'0').ljust(6,'0'))
        if len(cube_parts)>2:
            cube_version += int(cube_parts[2].rjust(2,'0').ljust(4,'0'))
        if len(cube_parts)>3:
            cube_version += int(cube_parts[3].rjust(2,'0').ljust(2,'0'))
        # for model
        model_version = 0        
        model_version += int(model_parts[0].rjust(2,'0').ljust(8,'0'))
        model_version += int(model_parts[1].rjust(2,'0').ljust(6,'0'))
        if len(model_parts)>2:
            model_version += int(model_parts[2].rjust(2,'0').ljust(4,'0'))
        if len(model_parts)>3:
            model_version += int(model_parts[3].rjust(2,'0').ljust(2,'0'))        
        return cube_version>=model_version        
    
    def info(self):
        """
        Prints available information 
        """
        print("Model info: ["+self.language_code+"-"+str(self.model_version)+"]")
        print("\tLanguage:                          "+self.language)
        print("\tShort language code:               "+self.language_code)
        print("\tVersion:                           "+str(self.model_version))
        print("\tMinimum NLP Cube required version: "+str(self.minimum_nlp_cube_version))
        print("\tUses space as token delimiter:     "+("True" if self.token_delimiter==" " else "False")   )
        print("\tEmbeddings file:                   "+self.embeddings_file_name)
        print("\tEmbeddings online link:            "+self.embeddings_remote_link)
        print("\tModel build date:                  "+self.model_build_date)
        print("\tModel build source:                "+self.model_build_source)
        print("\tNotes: "+self.notes)
        
class ModelStore(object):
    """
    Abstraction layer for working with models.
    """
    
    CLOUD_MODEL_REPO_LOCATION = 'https://raw.githubusercontent.com/adobe/NLP-Cube/master/MODEL_REPOSITORY'        
    
    def __init__(self, disk_path=None, cloud_path=None):        
        self.cloud_path = cloud_path #self.MODELS_PATH_CLOUD
        
        if disk_path == None:
            self.disk_path = os.path.join(str(Path.home()), ".nlpcube/models")
            if not os.path.exists(self.disk_path):
                os.makedirs(self.disk_path)        
        else:
            self.disk_path = disk_path

        self.embeddings_repository = os.path.join(self.disk_path, "embeddings")
        if not os.path.exists(self.embeddings_repository):
            os.makedirs(self.embeddings_repository)            
        
        self.model = {}
        self.metadata = ModelMetadata()        

    def _get_models_path_cloud (self):
        if self.cloud_path != None:
            return self.cloud_path 
        r = requests.get(self.CLOUD_MODEL_REPO_LOCATION, allow_redirects=True)
        return r.content.decode("utf-8").strip() # convert b'' to string
            
    def _list_folders (self, lang_code=None):              
        output = [os.path.basename(os.path.normpath(dI)) for dI in os.listdir(self.disk_path) if os.path.isdir(os.path.join(self.disk_path,dI))]
        if lang_code != None:
            output = [dI for dI in output if lang_code in dI]        
        return output        
        
    def find(self, lang_code, version="latest", verbose=True):
        """
        Contains logic for loading or downloading and loading models for the target language.
        
        Description: 
        if version == "latest":
            it checks for the local latest version available, and it loads it
            if it does not find any local version, it downloads the latest one it finds online
        if version == "2.0": (or any other specific version, != "latest")
            it checks for it locally, if it finds it, it loads it
            if it is not found locally, it attempts to download it from the cloud and then loads it
        Args:
            lang_code: Target language code.
                See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
            version: "latest" to get the latest version, or other specific version in like "1.0", "2.1", etc .
        """        
        # check for the latest local version, according to version parameter
        if version == "latest":
            latest_version = None
            local_models = self.list_local_models(lang_code)                     
            if len(local_models)>0:
                local_versions = [x[1] for x in local_models]
                local_versions.sort()
                latest_version = local_versions[-1]
                if verbose:
                    print("Loading latest local model: "+lang_code+"-"+str(latest_version))
                #self._load(lang_code,latest_version)                
                return os.path.join(self.disk_path,lang_code+"-"+str(latest_version))
            else: # no models found, check online according to version parameter
                if version=="latest":                    
                    version = self._version_to_download(lang_code, version=version)
                    if version!=None:
                        print("Latest version found online: "+lang_code+"-"+str(version))
                    else: # nothing was found online
                        raise Exception("No model version for language ["+lang_code+"] was found in the online repository!")                       
                self._download_model(lang_code, version)                
                return os.path.join(self.disk_path,lang_code+"-"+str(version)) #self._load(lang_code,version)
                
        else: # check for a specific local version, according to version parameter
            version = float(version)            
            if os.path.isdir(os.path.join(self.disk_path, lang_code +"-"+ str(version))):
                return os.path.join(self.disk_path,lang_code+"-"+str(version)) #self._load(lang_code,version)
            else: # version not found, trying to download it from the cloud
                cloud_version = self._version_to_download(lang_code, version=version)                
                if cloud_version == None:                
                    raise Exception("Version ["+str(version)+"] for language ["+lang_code+"] was not found in the online repository. Maybe try using .find(version='latest') to auto-download the latest model?")
                self._download_model(lang_code, str(cloud_version))                
                return os.path.join(self.disk_path,lang_code+"-"+str(cloud_version)) #self._load(lang_code,version)
        

    def _load(self, lang_code, version):
        """
        Load models on the class.
        """
        
        from cube.io_utils.encodings import Encodings
        from cube.io_utils.embeddings import WordEmbeddings
        from cube.io_utils.config import (TieredTokenizerConfig, CompoundWordConfig,
                             LemmatizerConfig, TaggerConfig, ParserConfig)
        from cube.generic_networks.tokenizers import TieredTokenizer
        from cube.generic_networks.token_expanders import CompoundWordExpander
        from cube.generic_networks.lemmatizers import FSTLemmatizer
        from cube.generic_networks.taggers import BDRNNTagger
        from cube.generic_networks.parsers import BDRNNParser
                
        # Refresh metadata
        self.metadata.read(os.path.join(self.disk_path,lang_code+"-"+str(version),"metadata.json"))
        model_folder = os.path.join(self.disk_path,lang_code+"-"+str(version))        
        embeddings_folder = os.path.join(self.disk_path,"embeddings")
        embeddings_file_path = os.path.join(embeddings_folder, self.metadata.embeddings_file_name)
        
        #sys.stdout.write('Loading model for {}-{}\n'.format(lang_code,version))
       
        # 1. Load word embeddings
        self.embeddings = WordEmbeddings(verbose=False)   
        sys.stdout.write('\tLoading embeddings ... \n')        
        self.embeddings.read_from_file(embeddings_file_path, None, full_load=False)

        # 2. Load tokenizer
        if not os.path.isfile(os.path.join(model_folder, 'tokenizer-tok.bestAcc')):
            sys.stdout.write('\tTokenization disabled. \n')
        else:
            self.tokenizer_enabled = True
            sys.stdout.write('\tTokenization enabled.\n')
            tokenizer_encodings = Encodings(verbose=False)
            tokenizer_encodings.load(os.path.join(model_folder, 'tokenizer.encodings'))
            config = TieredTokenizerConfig(os.path.join(model_folder, 'tokenizer.conf'))
            tokenizer_object = TieredTokenizer(config, tokenizer_encodings, self.embeddings, runtime=True)
            tokenizer_object.load(os.path.join(model_folder, 'tokenizer'))
            self.model[PipelineComponents.TOKENIZER] = tokenizer_object

        # 3. Load compound
        if not os.path.isfile(os.path.join(model_folder, 'compound.bestAcc')):
            sys.stdout.write('\tCompound disabled. \n')
        else:
            self.compound_enabled = True
            sys.stdout.write('\tCompound enabled.\n')
            compound_encodings = Encodings(verbose=False)
            compound_encodings.load(os.path.join(model_folder, 'compound.encodings'))
            config = CompoundWordConfig(os.path.join(model_folder, 'compound.conf'))
            compound_object = CompoundWordExpander(config, compound_encodings, self.embeddings, runtime=True)
            compound_object.load(os.path.join(model_folder, 'compound.bestAcc'))
            self.model[PipelineComponents.COMPOUND] = compound_object

        # 4. Load lemmatizer
        if not os.path.isfile(os.path.join(model_folder, 'lemmatizer.bestACC')):
            sys.stdout.write('\tLemmatizer disabled. \n')
        else:
            self.lemmatizer_enabled = True
            sys.stdout.write('\tLemmatizer enabled.\n')
            lemmatizer_encodings = Encodings(verbose=False)
            lemmatizer_encodings.load(os.path.join(model_folder, 'lemmatizer.encodings'))
            config = LemmatizerConfig(os.path.join(model_folder, 'lemmatizer.conf'))
            lemmatizer_object = FSTLemmatizer(config, lemmatizer_encodings, self.embeddings, runtime=True)
            lemmatizer_object.load(os.path.join(model_folder, 'lemmatizer.bestACC'))
            self.model[PipelineComponents.LEMMATIZER] = lemmatizer_object

        # 5. Load taggers
        if not os.path.isfile(os.path.join(model_folder, 'tagger.bestUPOS')):
            sys.stdout.write('\tTagger disabled. \n')
        else:
            self.tagger_enabled = True
            sys.stdout.write('\tTagger enabled.\n')
            tagger_encodings = Encodings(verbose=False)
            tagger_encodings.load(os.path.join(model_folder, 'tagger.encodings'))
            config = TaggerConfig(os.path.join(model_folder, 'tagger.conf'))

            tagger_upos_object = BDRNNTagger(config, tagger_encodings, self.embeddings, runtime=True)
            tagger_upos_object.load(os.path.join(model_folder, 'tagger.bestUPOS'))
            tagger_xpos_object = BDRNNTagger(config, tagger_encodings, self.embeddings, runtime=True)
            tagger_xpos_object.load(os.path.join(model_folder, 'tagger.bestXPOS'))
            tagger_attrs_object = BDRNNTagger(config, tagger_encodings, self.embeddings, runtime=True)
            tagger_attrs_object.load(os.path.join(model_folder, 'tagger.bestATTRS'))

            self.model[PipelineComponents.TAGGER] = [tagger_upos_object, tagger_xpos_object, tagger_attrs_object]
        
        # 6. Load parser
        if not os.path.isfile(os.path.join(model_folder, 'parser.bestUAS')):
            sys.stdout.write('\tParser disabled. \n')
        else:
            self.parser_enabled = True
            sys.stdout.write('\tParser enabled.\n')
            lemmatizer_encodings = Encodings(verbose=False)
            lemmatizer_encodings.load(os.path.join(model_folder, 'parser.encodings'))
            config = ParserConfig(os.path.join(model_folder, 'parser.conf'))
            parser_object = BDRNNParser(config, lemmatizer_encodings, self.embeddings, runtime=True)
            parser_object.load(os.path.join(model_folder, 'parser.bestUAS'))
            self.model[PipelineComponents.PARSER] = parser_object

    def import_local_model(self, model_file):
        """
        Import a local .zip file. Will overwrite folder.
        
        Args:
            @param model_file: The full path to the local .zip file        
        """
        if not os.path.exists(model_file):
            raise Exception("Model file not found: {}".format(model_file))
        
        print("Importing model {}".format(model_file))
        
        self._download_and_extract_model_zip(model_file, None)        
        self.metadata.read(os.path.join(self.disk_path,model_file.replace(".zip",""),"metadata.json"))
                
        # check embeddings url        
        if self.metadata.embeddings_remote_link != "": # don't download anything if link is not set
            import regex as re
            regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

            if re.match(regex, "http://www.example.com") is not None:
                # Download Facebook embeddings based on the metadata read from the model
                self._download_embeddings(self.metadata.embeddings_remote_link, self.metadata.embeddings_file_name)
            else: # some error in malformed URL probably
                raise Exception("The 'embeddings_remote_link' in the metadata file is malformed: ["+str(self.metadata.embeddings_remote_link)+"]")
                    
        print("\nModel {} was successfully imported.".format(model_file))
        
    def _download_model(self, lang_code, version):
        """
        Downloads pre-trained models for the provided language.

        Args:
            @param lang_code: Target language code.
                See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
            @param version: Version of the model.
        """
        
        model_name = '{}-{}'.format(lang_code, version)
        cloud_path = self._get_models_path_cloud()
        model_path_cloud = os.path.join(cloud_path, '{}.zip'.format(model_name))
        model_path_local = os.path.join(self.disk_path, '{}.zip'.format(model_name))
        
        # Download and extract models for provided language. 
        self._download_and_extract_model_zip(url=model_path_cloud, file_name=model_path_local)        
        self.metadata.read(os.path.join(self.disk_path,lang_code+"-"+str(version),"metadata.json"))
                
         # Download embeddings based on the metadata read from the model
        self._download_embeddings(self.metadata.embeddings_remote_link, self.metadata.embeddings_file_name)
        
        sys.stdout.write("\n")

    def _download_with_progress_bar(self, url, local_filename):
        r = requests.get(url, stream=True)
        total_size = int(r.headers['Content-Length'].strip())
        current_size = 0
        #request_content = []        
        with fopen(local_filename, 'wb') as f:
            for buf in r.iter_content(4096*16):            
                if buf:
                    #request_content.append(buf)
                    f.write(buf)
                    current_size += len(buf)  
                    done = int(40 * current_size / total_size)
                    sys.stdout.write("\r[%s%s] %3.1f%%, downloading %.2f/%.2f MB ..." % ('=' * done, ' ' * (40-done), 100* current_size/total_size, current_size/1024/1024, total_size/1024/1024) )    
                    sys.stdout.flush()
            #return b"".join(request_content)
        
        
    def _download_and_extract_model_zip(self, url, file_name, force=False):
        if file_name:
            if os.path.exists(file_name):
                if force:
                    os.remove(file_name)
                return
        
        temp_folder = tempfile.mkdtemp()  
        try:
            if file_name != None: # online zip file
            # Download and extract zip archive.
                zip_file_name = os.path.join(temp_folder, "tmp.zip")
                self._download_with_progress_bar(url, zip_file_name)       
                sys.stdout.write("\rDownload complete, decompressing files ...                                         ")
                sys.stdout.flush()
            else:
                zip_file_name = url
                        
            zipfile = ZipFile(zip_file_name, "r")
            zipfile.extractall(self.disk_path)
            zipfile.close()
            sys.stdout.write("\nModel extracted successfully.")
            sys.stdout.flush()
            
        except Exception as e:
            print("Error encountered, cleaning up and exiting ...")
            rmtree(temp_folder, ignore_errors=True)
            raise e
        
        # delete temporary folder
        rmtree(temp_folder, ignore_errors=True)

    def _download_embeddings(self, embeddings_remote_link, embeddings_file_name):
        """
        Download remote embeddings for the provided lang_code.
        Args:
            @param lang_code: Target language code.
                See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
            @param version: Version of the model to read which embedding file to get.
        """
        
        embeddings_folder = os.path.join(self.disk_path,"embeddings")
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)        
        embeddings_file = os.path.join(embeddings_folder,embeddings_file_name)
                
        # Check locally for the file
        sys.stdout.write("\nChecking for associated vector embeddings file ["+embeddings_file_name+"] ...\n")
        if os.path.isfile(embeddings_file):
            return         
            
        # We don't have the correct embedding file, download it ...        
        self._download_with_progress_bar(embeddings_remote_link, embeddings_file)                
        sys.stdout.write("\rEmbeddings downloaded successfully.                                                  ")        
    
    def _version_to_download(self, lang_code, version="latest"):
        """
        Returns the version of the language models that need to be downloaded,
        or None if there's nothing to be done.
        """        
        online_models = self.list_online_models(lang_code)

        # filter by lang code
        lang_models = [x for x in online_models if lang_code in x[0]]

        if len(lang_models)==0:
            return None # nothing found online
        
        if version == "latest":
            # Compute latest version.
            remote_versions = [x[1] for x in lang_models]
            remote_versions.sort()
            return remote_versions[-1]             
        else:
            for _, model_version in lang_models:                                
                if str(version) == str(model_version):
                    return version
            return None # not found this particular version online
    
    def delete_model(self, lang_code, version):
        """ 
        Deletes a local model. Also checks for associated embeddings file and cleans it up as well only if not referenced by any other local model
        """
        model = lang_code+"-"+str(version)
        model_folder = os.path.join(self.disk_path,model)
        # check if model exists
        if not os.path.isdir(model_folder):
            print("Model "+model+" not found! Nothing to delete.")
            return
            
        # determine which embedding file we need to delete
        model_metadata = ModelMetadata()                
        model_metadata.read(os.path.join(model_folder,"metadata.json"))        
        embeddings_file_to_delete = model_metadata.embeddings_file_name
        
        # delete the model
        import shutil
        try:
            shutil.rmtree(model_folder)
        except OSError as e:
            print ("Error removing folder from local disk: %s - %s." % (e.filename, e.strerror))
           
        # search in other models for referenced embeddings file
        found_in_other_models = False
        lang_models = self._list_folders() 
        for lang_model in lang_models:                        
            metadata_file_path = os.path.join(self.disk_path, lang_model, "metadata.json")
            if not os.path.exists(metadata_file_path):
                continue # this is not a model folder, so skip it            
            model_metadata.read(os.path.join(self.disk_path, lang_model, "metadata.json"))
            other_embeddings_file = model_metadata.embeddings_file_name
            if other_embeddings_file == embeddings_file_to_delete:
                found_in_other_models = True
                print("Embeddings file "+embeddings_file_to_delete+" is still being used by model "+ lang_model+" so it will not be deleted.")
                break
        if not found_in_other_models:
            try:
                embeddings_file_to_delete_abs_path = os.path.join(self.disk_path,"embeddings",embeddings_file_to_delete)
                os.remove(embeddings_file_to_delete_abs_path)
                print("Removed embeddings file "+embeddings_file_to_delete)
            except OSError as e:  ## if failed, report it back to the user ##
                print ("Error removing embeddings file: %s - %s." % (e.filename, e.strerror))
            
        print("Model cleanup successful.")
        
    def list_local_models(self, lang_code = None):
        """
        Returns a list of tuples of the models found locally
        ex: [("en",1.0),("en",1.1),("es",1.0)...]
        """
        lang_models = self._list_folders()         
        lang_models = [x for x in lang_models if x.count("-") == 1] # eliminate the embeddings and any other non-model folder                
        if len(lang_models)>0:            
            local_models = []
            for x in lang_models:
                try:
                    local_models.append((x.split("-")[0],float(x.split("-")[1])))
                except:
                    pass                         
            if lang_code:
                local_models = [x for x in local_models if lang_code in x[0]]
            return local_models
        else:
            return []    
        
    def list_online_models(self, lang_code = None):
        """
        Returns a list of tuples of the models found online
        ex: [("en",1.0),("en",1.1),("es",1.0)...]
        
        """        
        # get model store
        cloud_path = self._get_models_path_cloud()
        
        # now parse HTML page
        from bs4 import BeautifulSoup
        
        page = requests.get(cloud_path).text
        #print (page) # debug
        soup = BeautifulSoup(page, 'html.parser')
        online_models = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith('zip')]  

        """ # old version, interrogating azure repo
        request = requests.get(self.MODELS_PATH_CLOUD_ALL)
        data = xmltodict.parse(request.content)

        # Make a list with all the archives in the container.
        online_models = [item['Name']
                      for item in data['EnumerationResults']['Blobs']['Blob']
                      if item['Name'].endswith('.zip')]
        """
        online_models = [(x.replace(".zip","").split("-")[0],float(x.replace(".zip","").split("-")[1])) for x in online_models if "-" in x]
        if lang_code:
            online_models = [x for x in online_models if lang_code in x[0]]            
        return online_models
        
        
    def _copy_file(self, input_folder, output_folder, file_name):
        src_file = os.path.join(input_folder, file_name)
        dst_file = os.path.join(output_folder, file_name)
        if not os.path.isfile(src_file):            
            return False
        copyfile(src_file, dst_file)
        return True
            
    def _zipper(self, dir, zip_file):
        zip = zipfile.ZipFile(zip_file, 'w', compression=zipfile.ZIP_DEFLATED)
        root_len = len(os.path.abspath(dir))
        for root, dirs, files in os.walk(dir):
            archive_root = os.path.abspath(root)[root_len:]
            for f in files:
                fullpath = os.path.join(root, f)
                archive_name = os.path.join(archive_root, f)                
                zip.write(fullpath, archive_name, zipfile.ZIP_DEFLATED)
        zip.close()    
    
    def package_model(self, input_folder, output_folder_path, metadata, should_contain_tokenizer = True, should_contain_compound_word_expander = False, should_contain_lemmatizer = True, should_contain_tagger = True, should_contain_parser = True):
        """
            input_folder = "English-GWT"
            output_folder_path = "path_to_where_zip_files_will_be_placed"
        """
                
        # check input folder exists
        if not os.path.isdir(input_folder):
            raise Exception("Input folder not found")
            
        # create temporary folder locally            
        temp_folder = tempfile.mkdtemp()  
        try: 
            # create local model sub-folder
            output_folder = os.path.join(temp_folder, metadata.language_code+"-"+str(metadata.model_version))
            print("\tWriting model to temp folder: "+output_folder)
            os.makedirs(output_folder)        
            
            # write metadata to this folder
            metadata.save(os.path.join(output_folder,"metadata.json"))
            
            # copy tokenizer files        
            if should_contain_tokenizer:
                tokenizer_is_valid = True
                if not self._copy_file(input_folder, output_folder, "tokenizer.encodings"):
                    tokenizer_is_valid = False
                if not self._copy_file(input_folder, output_folder, "tokenizer.conf"):
                    tokenizer_is_valid = False
                if not self._copy_file(input_folder, output_folder, "tokenizer-tok.bestAcc"):
                    tokenizer_is_valid = False
                if not self._copy_file(input_folder, output_folder, "tokenizer-ss.bestAcc"):
                    tokenizer_is_valid = False            
                if tokenizer_is_valid:
                    print("\tTokenizer model found.")
                else:
                    raise Exception("Tokenizer model not found (or incomplete).")
            
            # copy compound_word_expander files        
            if should_contain_compound_word_expander:
                compound_word_expander = True
                if not self._copy_file(input_folder, output_folder, "compound.bestAcc"):
                    compound_word_expander = False
                if not self._copy_file(input_folder, output_folder, "compound.conf"):
                    compound_word_expander = False
                if not self._copy_file(input_folder, output_folder, "compound.encodings"):
                    compound_word_expander = False            
                if compound_word_expander:
                    print("\tCompound word expander model found.")
                else:
                    raise Exception("Compound word expander model not found (or incomplete).")
            
            # copy tagger files
            if should_contain_tagger:
                tagger = True
                if not self._copy_file(input_folder, output_folder, "tagger.bestUPOS"):
                    tagger = False
                if not self._copy_file(input_folder, output_folder, "tagger.bestXPOS"):
                    tagger = False
                if not self._copy_file(input_folder, output_folder, "tagger.bestATTRS"):
                    tagger = False
                if not self._copy_file(input_folder, output_folder, "tagger.conf"):
                    tagger = False
                if not self._copy_file(input_folder, output_folder, "tagger.encodings"):
                    tagger = False            
                if tagger:
                    print("\tTagger model found.")
                else:
                    raise Exception("Tagger model not found (or incomplete).")
            
            # copy lemmatizer files
            if should_contain_lemmatizer:
                lemmatizer = True
                # patch 
                if os.path.isfile(os.path.join(input_folder, "lemmatizer.bestACC")):
                    os.rename(os.path.join(input_folder, "lemmatizer.bestACC"), os.path.join(input_folder, "lemmatizer.bestAcc"))
                if not self._copy_file(input_folder, output_folder, "lemmatizer.bestAcc"):
                    lemmatizer = False
                if not self._copy_file(input_folder, output_folder, "lemmatizer.conf"):
                    lemmatizer = False
                if not self._copy_file(input_folder, output_folder, "lemmatizer.encodings"):
                    lemmatizer = False            
                if lemmatizer:
                    print("\tLemmatizer model found.")
                else:
                    raise Exception("Lemmatizer model not found (or incomplete).")
            
            # copy parser files
            if should_contain_parser:
                parser = True
                if not self._copy_file(input_folder, output_folder, "parser.bestUAS"):
                    parser = False
                if not self._copy_file(input_folder, output_folder, "parser.bestLAS"):
                    parser = False
                if not self._copy_file(input_folder, output_folder, "parser.conf"):
                    parser = False
                if not self._copy_file(input_folder, output_folder, "parser.encodings"):
                    parser = False            
                if parser:
                    print("\tParser model found.")
                else:
                    raise Exception("Parser model not found (or incomplete).")
            
            # package into zip file
            print("\tCompressing model ...")    
            
            model_file = os.path.join(output_folder_path,metadata.language_code+"-"+str(metadata.model_version)+".zip")            
            self._zipper(temp_folder, model_file)
            
        except Exception as e:
            print("Error encountered, cleaning up and exiting ...")
            rmtree(temp_folder, ignore_errors=True)
            raise e
        
        # delete temporary folder
        print("\tCleaning up ...")
        rmtree(temp_folder, ignore_errors=True)
        
        print("Model packaged successfully as: "+model_file)