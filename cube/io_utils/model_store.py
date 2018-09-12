#
# Author: Ruxandra Burtica
#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import defaultdict
import io
import os
import sys
import json
from misc.misc import fopen
from zipfile import ZipFile

import requests
import xmltodict

# Append parent dir to sys path.
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

from io_utils.encodings import Encodings
from io_utils.embeddings import WordEmbeddings
from io_utils.config import (TieredTokenizerConfig, CompoundWordConfig,
                     LemmatizerConfig, TaggerConfig, ParserConfig)
from generic_networks.tokenizers import TieredTokenizer
from generic_networks.token_expanders import CompoundWordExpander
from generic_networks.lemmatizers import FSTLemmatizer
from generic_networks.taggers import BDRNNTagger
from generic_networks.parsers import BDRNNParser
   
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
        # OPTIONAL: model build date: string
        self.model_build_date = ""
        # OPTIONAL: model build source: what corpus was it built from. Ex: UD-Romanian-RRT v2.2 
        self.model_build_source = ""
        # OPTIONAL: other notes, string value
        self.notes = ""
    
    def read(self, filename):            
        assert(filename.endswith("metadata.json"))
        data = json.load(fopen(filename,"r"))
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
        obj["embeddings_remote_link"] = self.embeddings_remote_link
        obj["embeddings_file_name"] = self.embeddings_file_name
        obj["token_delimiter"] = self.token_delimiter
        obj["model_build_date"] = self.model_build_date
        obj["model_build_source"] = self.model_build_source
        obj["notes"] = self.notes
        json.dump(obj, fopen(filename,"w"), indent=4, sort_keys=True)    
        
    def info(self):
        """
        Prints available information 
        """
        print("Model info: ["+self.language_code+"-"+str(self.model_version)+"]")
        print("\tLanguage:                      "+self.language)
        print("\tShort language code:           "+self.language_code)
        print("\tVersion:                       "+str(self.model_version))
        print("\tUses space as token delimiter: "+("True" if self.token_delimiter==" " else "False")   )
        print("\tEmbeddings file:               "+self.embeddings_file_name)
        print("\tEmbeddings online link:        "+self.embeddings_remote_link)
        print("\tModel build date:              "+self.model_build_date)
        print("\tModel build source:            "+self.model_build_source)
        print("\tNotes: "+self.notes)
        
class ModelStore(object):
    """
    Abstraction layer for working with models.
    """

    MODELS_PATH_LOCAL = 'models'
    MODELS_PATH_CLOUD = 'https://nlpcube.blob.core.windows.net/models'
    MODELS_PATH_CLOUD_ALL = os.path.join(MODELS_PATH_CLOUD, '?restype=container&comp=list')

    def __init__(self, disk_path=None, cloud_path=None):
        self.disk_path = disk_path or self.MODELS_PATH_LOCAL
        self.cloud_path = cloud_path or self.MODELS_PATH_CLOUD
        self.model = {}
        self.metadata = ModelMetadata()

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
            if os.path.isdir(os.path.join(self.disk_path, lang_code, version)):
                return os.path.join(self.disk_path,lang_code+"-"+str(version)) #self._load(lang_code,version)
            else: # version not found, trying to download it from the cloud
                version = self._version_to_download(lang_code, version=version)                
                if version == None:                
                    raise Exception("Version ["+str(version)+"] for language ["+lang_code+"] was not found in the online repository. Maybe try using .find(version='latest') to auto-download the latest model?")
                self._download_model(lang_code, str(version))                
                return os.path.join(self.disk_path,lang_code+"-"+str(version)) #self._load(lang_code,version)
        

    def _load(self, lang_code, version):
        """
        Load models on the class.
        """
        # Refresh metadata
        self.metadata.read(os.path.join(self.disk_path,lang_code+"-"+str(version),"metadata.json"))
        model_folder = os.path.join(self.disk_path,lang_code+"-"+str(version))        
        embeddings_folder = os.path.join(self.disk_path,"embeddings")
        embeddings_file_path = os.path.join(embeddings_folder, self.metadata.embeddings_file_name)
        
        #sys.stdout.write('Loading model for {}-{}\n'.format(lang_code,version))
       
        # 1. Load word embeddings
        self.embeddings = WordEmbeddings(verbose=False)   
        sys.stdout.write('\tLoading embeddings... \n')        
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

    def _download_model(self, lang_code, version):
        """
        Downloads pre-trained models for the provided language.

        Args:
            @param lang_code: Target language code.
                See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
            @param version: Version of the model.
        """
        #sys.stdout.write('Downloading models for {} \n'.format(lang_code))

        model_name = '{}-{}'.format(lang_code, version)
        model_path_cloud = os.path.join(self.cloud_path, '{}.zip'.format(model_name))
        model_path_local = os.path.join(self.disk_path, '{}.zip'.format(model_name))
        
        # Download and extract models for provided language. 
        self._download_and_extract_lang_model(model_path_cloud, model_path_local)        
        self.metadata.read(os.path.join(self.disk_path,lang_code+"-"+str(version),"metadata.json"))
        
        # Download Facebook embeddings based on the metadata read from the model
        self._download_embeddings(self.metadata.embeddings_remote_link, self.metadata.embeddings_file_name)
        sys.stdout.write("\n")

    def _download_with_progress_bar(self, url):
        r = requests.get(url, stream=True)
        total_size = int(r.headers['Content-Length'].strip())
        current_size = 0
        request_content = []        
        for buf in r.iter_content(4096*16):            
            if buf:
                request_content.append(buf)
                current_size += len(buf)  
                done = int(40 * current_size / total_size)
                sys.stdout.write("\r[%s%s] %3.1f%%, downloading %.2f/%.2f MB ..." % ('=' * done, ' ' * (40-done), 100* current_size/total_size, current_size/1024/1024, total_size/1024/1024) )    
                sys.stdout.flush()
        return b"".join(request_content)
        
    def _download_and_extract_lang_model(self, url, file_name, force=False):
        if os.path.exists(file_name):
            if force:
                os.remove(file_name)
            return

        # Download and extract zip archive.
        request_content = self._download_with_progress_bar(url)
        #request = requests.get(url)
        #request_content = request.content        
        sys.stdout.write("\rDownload complete, decompressing files ...                                         ")
        sys.stdout.flush()
        
        zipfile = ZipFile(io.BytesIO(request_content))
        zipfile.extractall(self.disk_path)
        zipfile.close()
        sys.stdout.write("\nModel downloaded successfully.")
        sys.stdout.flush()

    def _download_embeddings(self, embeddings_remote_link, embeddings_file_name):
        """
        Download remote embeddings for the provided lang_code.
        Args:
            @param lang_code: Target language code.
                See http://opensource.adobe.com/NLP-Cube/ for available languages and their codes
            @param version: Version of the model to read which embedding file to get.
        """
        
        embeddings_folder = os.path.join(self.disk_path,"embeddings")
        embeddings_file = os.path.join(embeddings_folder,embeddings_file_name)
                
        # Check locally for the file
        sys.stdout.write("\nChecking for associated vector embeddings file ["+embeddings_file_name+"] ...\n")
        if os.path.isfile(embeddings_file):
            return 
        
        # We don't have the correct embedding file, download it ...        
        request_content = self._download_with_progress_bar(embeddings_remote_link)
        sys.stdout.write("\rDownload complete, flushing to disk ...                                              ")
        sys.stdout.flush()        
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)
        with open(embeddings_file, 'wb') as fd:
            fd.write(request_content)
        sys.stdout.write("\nEmbeddings downloaded successfully.")        
    
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
            for model in lang_models:
                if str(version) in model:
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
        model_metadata.load(lang_code+"-"+str(version))
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
            model_metadata.load(lang_model)
            other_embeddings_file = model_metadata.embeddings_file_name
            if other_embeddings_file == embeddings_file_to_delete:
                found_in_other_models = True
                print("Embeddings file "+embeddings_file_to_delete+" is still being used by model "+ lang_model+" so it will not be deleted.")
                break
        if not found_in_other_models:
            try:
                os.remove(embeddings_file_to_delete)
                print("Removed embeddings file "+embeddings_file_to_delete+".")
            except OSError as e:  ## if failed, report it back to the user ##
                print ("Error removing embeddings file: %s - %s." % (e.filename, e.strerror))
            
        print("Model cleanup successful.")
        
    def list_local_models(self, lang_code = None):
        """
        Returns a list of tuples of the models found locally
        ex: [("en",1.0),("en",1.1),("es",1.0)...]
        """
        lang_models = self._list_folders()         
        lang_models = [x for x in lang_models if "-" in x] # eliminate the embeddings and any other non-model folder        
        if len(lang_models)>0:            
            local_models = [(x.split("-")[0],float(x.split("-")[1])) for x in lang_models]            
            if lang_code:
                local_models = [x for x in local_models if lang_code in x[0]]
            return local_models
        else:
            return []    
        
    def list_online_models(self, lang_code):
        """
        Returns a list of tuples of the models found online
        ex: [("en",1.0),("en",1.1),("es",1.0)...]
        """
        request = requests.get(self.MODELS_PATH_CLOUD_ALL)
        data = xmltodict.parse(request.content)

        # Make a list with all the archives in the container.
        online_models = [item['Name']
                      for item in data['EnumerationResults']['Blobs']['Blob']
                      if item['Name'].endswith('.zip')]
        online_models = [(x.replace(".zip","").split("-")[0],float(x.replace(".zip","").split("-")[1])) for x in online_models if "-" in x]
        if lang_code:
            online_models = [x for x in online_models if lang_code in x[0]]            
        return online_models