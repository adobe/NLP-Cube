import os, sys, logging, json, requests, uuid, ntpath
from pathlib import Path
from typing import Optional
from tqdm.autonotebook import tqdm as tqdm
from cube.data.components import Component, TokenizerComponent, MWExpanderComponent, POSTaggerComponent, LemmatizerComponent, ParserComponent, NERComponent

class ModelStore ():
    root_path = os.path.join(str(Path.home()), ".nlpcube")

    @staticmethod
    def solve (lang:str, components:Optional[str]) -> list[Component]:
        """
        TODO docs
        :param lang: Language code
        :param components: string with comma separated components or None. If None, will get default components for the specified lanugage
        :return: List of Component objects
        """
        component_objects = []

        # check local catalog and download if not present
        catalog = ModelStore._get_catalog()
        catalog = catalog["default"] # for future use
        model_url = catalog["URL"]

        for component_str in components:
            logging.debug("ModelStore solving {}...".format(component_str))

            # check in catalog for component type
            if component_str == "tokenize":
                catalog_section = catalog["Tokenizer"]
                component = TokenizerComponent()
            elif component_str == "tokenizer":
                catalog_section = catalog["MWExpander"]
                component = MWExpanderComponent()
            elif component_str == "tokenizer":
                catalog_section = catalog["POSTagger"]
                component = POSTaggerComponent()
            elif component_str == "tokenizer":
                catalog_section = catalog["Lemmatizer"]
                component = LemmatizerComponent()
            elif component_str == "parse":
                catalog_section = catalog["Parser"]
                component = ParserComponent()
            elif component_str == "ner":
                catalog_section = catalog["NER"]
                component = NERComponent()
            else:
                raise Exception("Component '{}' not found in catalog!".format(component_str))

            # get model filename
            model_filename = None
            for possible_model_filename in catalog_section:
                if lang in catalog_section[possible_model_filename]:
                    model_filename = possible_model_filename
                    break
            if not model_filename:
                raise Exception("Model for component '{}' not found online!".format(component_str))

            # if model is not locally stored, retrieve model from online store and save locally
            model_filepath = os.path.join(ModelStore.root_path,"models",model_filename)
            if not os.path.exists(model_filepath):
                ModelStore._get_model(model_url, model_filename, progress_bar_position=-1)
            if not os.path.exists(model_filepath):
                raise Exception("Sanity check failed, model {} was not found locally!".format(model_filepath))

            # save filepath in component
            component.model_filepath = model_filepath

            # append to returned list of components
            component_objects.append(component)

        return component_objects

    @staticmethod
    def _get_catalog(url:str):
        local_path = os.path.join(ModelStore.root_path, "catalog.json")

        if not os.path.exits(local_path):
            logging.info("Catalog does not exists, downloading ... ")
            status_code, response_text = ModelStore.__download_file(url, local_path)
            if status_code != 200:
                raise Exception("Catalog download failed with status_code {}, message: ".format(status_code, response_text))

        if not os.path.exits(local_path):
            raise Exception("Sanity check failed, catalog file not found locally!")

        with open(local_path, "r", encoding="utf8") as f:
            catalog = json.load(f)

        return catalog

    @staticmethod
    def _get_model(model_url:str, model_filename:str, progress_bar_position:int=-1):
        try:
            file_counter = 1
            temp_folder = os.path.join(ModelStore.root_path, str(uuid.uuid4().hex))
            os.mkdir(temp_folder)

            # download files until we get a 404
            while True:
                current_file = os.path.join(temp_folder, "{}.{}".format(model_filename,file_counter))
                current_url = "{}{}.{}".format(model_url, model_filename, file_counter)
                logging.info("Fetching {} ...".format(current_url))
                # no more files to download
                if ModelStore.__download_file(current_url, current_file, progress_bar_position) != 200:
                    break
                else:
                    file_counter += 1

            if file_counter == 1:
                raise Exception("Could not download model!")

            # merge files
            with open(os.path.join(ModelStore.root_path,"models",model_filename), "wb") as f:
                for i in range(1, file_counter+1):
                    with open(os.path.join(temp_folder, "{}.{}".format(model_filename,i)), "rb") as r:
                        f.write(r.read())

        except Exception as ex:
            logging.error(str(ex))
        finally:
            if os.path.exists(temp_folder):
                os.rmdir(temp_folder)


    @staticmethod
    def __download_file(url:str, filename:str, progress_bar_position:int=-1):
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            return r.status_code
        total_size = int(r.headers['Content-Length'].strip())
        current_size = 0
        # request_content = []
        with open(filename, 'wb') as f:
            for buf in r.iter_content(4096 * 16):
                if buf:
                    f.write(buf)
                    current_size += len(buf)
                    done = int(40 * current_size / total_size)
                    sys.stdout.write("\r[%s%s] %3.1f%%, downloading %.2f/%.2f MB ..." % (
                    '=' * done, ' ' * (40 - done), 100 * current_size / total_size, current_size / 1024 / 1024,
                    total_size / 1024 / 1024))
                    sys.stdout.flush()
        return r.status_code

    @staticmethod
    def _pack_model(file_path:str, output_folder:str):
        logging.info("Packing {} to folder {}:".format(file_path, output_folder))
        _, filename = ntpath.split(file_path)
        counter = 1
        with open(file_path,"rb") as f:
            while True:
                byte_s = f.read(50*1024*1024) # 50MB chunks
                if not byte_s:
                    break
                chunk_file_path = os.path.join(output_folder, "{}.{}".format(filename,counter))
                logging.info("\t writing {:.2f}MB to {} ...".format(len(byte_s)/1024/1024, chunk_file_path))
                with open(chunk_file_path, "wb") as r:
                    r.write(byte_s)
                counter += 1


# ensure we get a valid root path for local model storage
ModelStore.root_path = os.path.join(str(Path.home()), ".nlpcube")
if not os.path.exists(ModelStore.root_path):
    os.mkdirs(ModelStore.root_path, exist_ok=True)