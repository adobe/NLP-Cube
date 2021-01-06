import os, sys, json, logging, requests, uuid, ntpath, traceback
import zipfile

parent_dir = os.path.dirname(os.path.abspath(__file__))
os.sys.path.insert(0, parent_dir)

from pathlib import Path
from shutil import copyfile, rmtree
from typing import Optional, List
from tqdm.autonotebook import tqdm as tqdm
from cube.io_utils.components import Component, TokenizerComponent, MWExpanderComponent, POSTaggerComponent, LemmatizerComponent, ParserComponent, NERComponent
logger = logging.getLogger('cube')

class ModelStore ():
    root_path = os.path.join(str(Path.home()), ".nlpcube2")

    @staticmethod
    def solve (lang:str, check_for_updates:bool = False) -> dict:
        component_objects = []

        # check local catalog and download if not present
        catalog = ModelStore._get_catalog(url = "https://raw.githubusercontent.com/adobe/NLP-Cube-Models/2.0/catalog.json")
        catalog = catalog["default"]

        # get model filename
        model_name = None
        for language_family in catalog:
            if lang in language_family["languages"]:
                model_url = language_family["model"]
                model_name = model_url[model_url.rfind("/")+1:]
                break
        if not model_name:
            raise Exception("Model for language '{}' not found in catalog. Please check on the official site for support for this language.".format(lang))

        # if model is not locally stored, retrieve model from online store and save locally
        model_folder = os.path.join(ModelStore.root_path, "models", model_name)
        if not os.path.isdir(model_folder):
            ModelStore._get_model(model_url, model_name)

        # TODO additional check if everything exists okay

        # return paths
        paths = {}

        # tokenizer paths
        if os.path.exists(os.path.join(ModelStore.root_path, "models", model_name, "tokenizer-{}.bestTOK".format(model_name))):
            paths["tokenizer"] = {
                "model" : os.path.join(ModelStore.root_path, "models", model_name, "tokenizer-{}.bestTOK".format(model_name)),
                "config" : os.path.join(ModelStore.root_path, "models", model_name, "tokenizer-{}.conf".format(model_name)),
                "encodings" : os.path.join(ModelStore.root_path, "models", model_name, "tokenizer-{}.encodings".format(model_name)),
            }

        # mwe paths
        if os.path.exists(os.path.join(ModelStore.root_path, "models", model_name, "compound-{}.bestTOK".format(model_name))):
            paths["compound"] = {
                "model" : os.path.join(ModelStore.root_path, "models", model_name, "compound-{}.bestTOK".format(model_name)),
                "config" : os.path.join(ModelStore.root_path, "models", model_name, "compound-{}.conf".format(model_name)),
                "encodings" : os.path.join(ModelStore.root_path, "models", model_name, "compound-{}.encodings".format(model_name)),
            }

        # lemmatizer paths
        if os.path.exists(os.path.join(ModelStore.root_path, "models", model_name, "lemmatizer-{}.best".format(model_name))):
            paths["lemmatizer"] = {
                "model" : os.path.join(ModelStore.root_path, "models", model_name, "lemmatizer-{}.best".format(model_name)),
                "config" : os.path.join(ModelStore.root_path, "models", model_name, "lemmatizer-{}.conf".format(model_name)),
                "encodings" : os.path.join(ModelStore.root_path, "models", model_name, "lemmatizer-{}.encodings".format(model_name)),
            }

        # tagger paths
        if os.path.exists(os.path.join(ModelStore.root_path, "models", model_name, "tagger-{}.bestUPOS".format(model_name))):
            paths["tagger"] = {
                "model_UPOS" : os.path.join(ModelStore.root_path, "models", model_name, "tagger-{}.bestUPOS".format(model_name)),
                "model_XPOS" : os.path.join(ModelStore.root_path, "models", model_name, "tagger-{}.bestXPOS".format(model_name)),
                "model_ATTRS" : os.path.join(ModelStore.root_path, "models", model_name, "tagger-{}.bestATTRS".format(model_name)),
                "config" : os.path.join(ModelStore.root_path, "models", model_name, "tagger-{}.conf".format(model_name)),
                "encodings" : os.path.join(ModelStore.root_path, "models", model_name, "tagger-{}.encodings".format(model_name)),
            }

        # parser paths
        if os.path.exists(os.path.join(ModelStore.root_path, "models", model_name, "parser-{}.bestUAS".format(model_name))):
            paths["parser"] = {
                "model": os.path.join(ModelStore.root_path, "models", model_name, "parser-{}.bestUAS".format(model_name)),
                "config" : os.path.join(ModelStore.root_path, "models", model_name, "parser-{}.conf".format(model_name)),
                "encodings" : os.path.join(ModelStore.root_path, "models", model_name, "parser-{}.encodings".format(model_name)),
            }

        return paths

    @staticmethod
    def _get_catalog(url:str):
        # TODO implement check_for_updates
        local_path = os.path.join(ModelStore.root_path, "catalog.json")

        if not os.path.exists(local_path):
            logger.info("Catalog does not exist locally, downloading ... ")
            status_code = ModelStore.__download_file(url+"?raw=true", local_path, disable_pbar=True)
            if status_code != 200:
                raise Exception("Catalog download failed with status_code {}".format(status_code))

        if not os.path.exists(local_path):
            raise Exception("Sanity check failed, catalog file not found locally!")

        with open(local_path, "r", encoding="utf8") as f:
            catalog = json.load(f)

        return catalog

    @staticmethod
    def _get_model(model_url:str, model_name:str, check_for_updates:bool = False):
        try:
            file_counter = 1
            temp_folder = os.path.join(ModelStore.root_path, str(uuid.uuid4().hex))
            os.mkdir(temp_folder)

            # download files until we get a 404
            while True:
                current_file = os.path.join(temp_folder, "{}.{}".format(model_name,file_counter))
                current_url = "{}.{}?raw=true".format(model_url, file_counter)
                #logger.info("Fetching model part {} from {} ...".format(file_counter, current_url))
                # no more files to download
                desc = "Downloading model part {}".format(file_counter)
                if ModelStore.__download_file(current_url, current_file, desc = desc) != 200:
                    break
                else:
                    file_counter += 1

            file_counter -= 1
            if file_counter == 0:
                raise Exception("Could not download model!")

            # merge files
            logger.info("Model {} successfully downloaded, unpacking".format(model_name))
            zip_file_name = os.path.join(temp_folder,model_name+".zip")
            with open(zip_file_name, "wb") as f:
                for i in range(1, file_counter+1):
                    with open(os.path.join(temp_folder, "{}.{}".format(model_name,i)), "rb") as r:
                        f.write(r.read())

            # unzip files
            output_folder = os.path.join(ModelStore.root_path, "models", model_name)
            ModelStore._unpack_model(zip_file_name = zip_file_name, output_folder = output_folder)

            return output_folder

        except Exception as ex:
            logger.error(traceback.format_exc())
        finally:
            if os.path.exists(temp_folder):
                rmtree(temp_folder, ignore_errors=True)
            return None

    @staticmethod
    def __download_file(url:str, filename:str, disable_pbar:bool=False, desc:str=None):
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            return r.status_code
        total_size = int(r.headers['Content-Length'].strip())
        pbar = tqdm(total = int(total_size/1000), unit="KB", leave=None, desc = desc, ncols = 120, disable=disable_pbar)
        current_size = 0

        sys.stdout.flush()
        sys.stderr.flush()

        with open(filename, 'wb') as f:
            for buf in r.iter_content(100000):
                if buf:
                    f.write(buf)
                    current_size += len(buf)
                    pbar.n = int(current_size/1000)
                    pbar.refresh()


        pbar.close()

        sys.stdout.flush()
        sys.stderr.flush()

        return r.status_code

    @staticmethod
    def _pack_model(input_folder: List[str], output_file_path: str):
        logger.info("Packing folder '{}' ".format(input_folder))
        if not os.path.isdir(input_folder):
            raise Exception("Folder '{}' not found.".format(input_folder))
        zip = zipfile.ZipFile(output_file_path, 'w', compression=zipfile.ZIP_DEFLATED)
        for f in os.listdir(input_folder):
            if f.endswith(".last"):
                continue
            filepath = os.path.join(input_folder, f)
            if os.path.isfile(filepath):
                logger.info("Adding '{}' to zip file ...".format(filepath))
                zip.write(filepath, f, zipfile.ZIP_DEFLATED)
        zip.close()
        logger.info("Done.")

    @staticmethod
    def _split_packed_model(file_path:str, output_folder:str):
        logger.info("Splitting {} to folder {}:".format(file_path, output_folder))
        _, filename = ntpath.split(file_path)
        counter = 1
        with open(file_path,"rb") as f:
            while True:
                byte_s = f.read(50*1000*1000) # 50MB chunks
                if not byte_s:
                    break
                chunk_file_path = os.path.join(output_folder, "{}.{}".format(filename.replace(".zip",""),counter))
                logger.info("\t writing {:.2f}MB to {} ...".format(len(byte_s)/1000/1000, chunk_file_path))
                with open(chunk_file_path, "wb") as r:
                    r.write(byte_s)
                counter += 1
        logger.info("Done.")

    @staticmethod
    def _join_packed_model(input_file_path: str, output_folder: str):
        pass

    @staticmethod
    def _unpack_model(zip_file_name: str, output_folder: str):
        logger.info("Unpacking '{}' to '{}'".format(zip_file_name, output_folder))
        os.makedirs(output_folder, exist_ok=True)
        zip = zipfile.ZipFile(zip_file_name, "r")
        zip.extractall(output_folder)
        zip.close()

    @staticmethod
    def _copy_file(input_folder, output_folder, file_name):
        src_file = os.path.join(input_folder, file_name)
        dst_file = os.path.join(output_folder, file_name)
        if not os.path.isfile(src_file):
            return False
        copyfile(src_file, dst_file)
        return True


# ensure we get a valid root path for local model storage
try:
    ModelStore.root_path = os.path.join(str(Path.home()), ".nlpcube2")
    if not os.path.exists(ModelStore.root_path):
        os.makedirs(ModelStore.root_path, exist_ok=True)
except Exception as ex:
    logger.error("Could not find model store / could not create path in '{}', failed with exception: {}".format(ModelStore.root_path, str(ex)))