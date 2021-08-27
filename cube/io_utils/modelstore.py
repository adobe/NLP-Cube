import os, sys, logging, json, requests, uuid, shutil
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm.autonotebook import tqdm as tqdm

logger = logging.getLogger('cube')

class ModelStore:
    """
    The purpose of this class is to be called from the api.load(language) and give back a list of component paths,
    or download them if they are not locally stored

    """
    root_path = os.path.join(str(Path.home()), ".nlpcube")
    catalog_url = "https://raw.githubusercontent.com/adobe/NLP-Cube-Models/3.0/models/catalog.json"

    @staticmethod
    def solve(lang: str, version: str = "latest", check_for_latest: bool = False) -> Tuple[dict, int]:
        """
        TODO docs
        :param lang: Language
        :param version: force particular version, else latest
        :return: Dict of paths, language id
        """
        paths = {
            "tokenizer": {},
            "cwe": {},
            "lemmatizer": {},
            "tagger": {},
            "parser": {}
        }

        # check local catalog and download if not present
        catalog = ModelStore._get_catalog(check_for_latest = check_for_latest)

        # identify all entries that match this lang and select the appropriate one
        if lang not in catalog:
            raise Exception(f"Language '{lang}' is not available!")

        entries = catalog[lang]
        if len(entries) == 0:
            raise Exception(f"Language '{lang}' is not available!")

        found = False
        if version != "latest": # check for specific version
            for entry in entries:
                if entry["version"] == version:
                    found = True
                    model_url = entry["link"]
                    langid = entry["langid"]
                    parts = entry["parts"]
        else:
            max_version = 0.0
            for entry in entries: # check for latest version
                try: # get version
                    entry_version = float(entry["version"])
                    if entry_version > max_version:
                        found = True
                        max_version = entry_version
                        model_url = entry["link"]
                        langid = entry["langid"]
                        parts = entry["parts"]
                except:
                    pass
            version = max_version

        if not found:
            raise Exception(f"Language '{lang}', version '{version}' not found, but there are other versions available!")

        # check if it's present, else download model in folder "name.version", return abspath of model folder
        model_folder = ModelStore._get_model(model_url, parts)

        # compose and return paths
        files = [x for x in os.listdir(model_folder)] # file names only

        #files = [os.path.abspath(os.path.join(model_folder, x)) for x in files]  # get full path

        tokenizer_files = [x for x in files if "tokenizer" in x]
        cwe_files = [x for x in files if "cwe" in x]
        lemmatizer_files = [x for x in files if "lemmatizer" in x]
        tagger_files = [x for x in files if "tagger" in x]
        parser_files = [x for x in files if "parser" in x]

        if len(tokenizer_files) > 0:
            tokenizer_entry = {"config": ModelStore.__get_file_path(tokenizer_files, ".config", model_folder),
                               "encodings": ModelStore.__get_file_path(tokenizer_files, ".encodings", model_folder),
                               #"sent": ModelStore.__get_file_path(tokenizer_files, ".sent", model_folder),
                               "model": ModelStore.__get_file_path(tokenizer_files, ".tok", model_folder)}

            if tokenizer_entry["config"] and tokenizer_entry["encodings"] and tokenizer_entry["model"]:
                paths["tokenizer"] = tokenizer_entry

        if len(cwe_files) > 0:
            cwe_entry = {"config": ModelStore.__get_file_path(cwe_files, ".config", model_folder),
                         "encodings": ModelStore.__get_file_path(cwe_files, ".encodings", model_folder),
                         "model": ModelStore.__get_file_path(cwe_files, ".best", model_folder)}

            if cwe_entry["config"] and cwe_entry["encodings"] and cwe_entry["model"]:
                paths["cwe"] = cwe_entry

        if len(lemmatizer_files) > 0:
            lemmatizer_entry = {"config": ModelStore.__get_file_path(lemmatizer_files, ".config", model_folder),
                                "encodings": ModelStore.__get_file_path(lemmatizer_files, ".encodings", model_folder),
                                "model": ModelStore.__get_file_path(lemmatizer_files, ".best", model_folder)}

            if lemmatizer_entry["config"] and lemmatizer_entry["encodings"] and lemmatizer_entry["model"]:
                paths["lemmatizer"] = lemmatizer_entry

        """
        if len(tagger_files) > 0:
            tagger_entry = {"config": ModelStore.__get_file_path(tagger_files, ".config", model_folder),
                            "encodings": ModelStore.__get_file_path(tagger_files, ".encodings", model_folder),
                            "sent": ModelStore.__get_file_path(tagger_files, ".sent", model_folder),
                            "tok": ModelStore.__get_file_path(tagger_files, ".tok", model_folder)}

            if tagger_entry["config"] and tagger_entry["encodings"] and tagger_entry["sent"] and \
                    tagger_entry["tok"]:
                paths["tagger"] = tagger_entry
        """

        if len(parser_files) > 0:
            parser_entry = {"config": ModelStore.__get_file_path(parser_files, ".config", model_folder),
                            "encodings": ModelStore.__get_file_path(parser_files, ".encodings", model_folder),
                            "model": ModelStore.__get_file_path(parser_files, ".las", model_folder)}

            if parser_entry["config"] and parser_entry["encodings"] and parser_entry["model"]:
                paths["parser"] = parser_entry

        return paths, langid

    @staticmethod
    def __get_file_path(files, extension, model_folder):
        """
        This function returns the abspath of the only one file from the 'files' list that has the given extension.
        'files' does not contain paths, only filenames
        """
        valid_files = []
        for file in files:
            if file.endswith(extension):
                valid_files.append(file)
        if len(valid_files)!=1:
            return None
        return os.path.abspath(os.path.join(model_folder, valid_files[0]))


    @staticmethod
    def _get_catalog(check_for_latest: bool = False):
        local_path = os.path.join(ModelStore.root_path, "catalog.json")

        if not os.path.exists(local_path) or check_for_latest is True:
            print("Catalog either does not exist or looking for updates, downloading ... ")
            status_code = ModelStore.__download_file(ModelStore.catalog_url, local_path)
            if status_code != 200:
                raise Exception(
                    "Catalog download failed with status_code {}".format(status_code))

        if not os.path.exists(local_path):
            raise Exception("Sanity check failed, catalog file not found locally, though it was downloaded!")

        with open(local_path, "r", encoding="utf8") as f:
            catalog = json.load(f)

        return catalog

    @staticmethod
    def _get_model(model_url: str, parts: int):
        if model_url.endswith("/"): # remove last /,
            model_url = model_url[:-1]

        model_name = model_url.split("/")[-1] # e.g. en_partut-1.0
        model_folder = os.path.join(ModelStore.root_path, "models", model_name)

        # check if model already exists
        if os.path.exists(model_folder):
            if len([f for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f))])>0:
                return os.path.abspath(model_folder) # model is present
                # todo Sanity check that model is valid here, not in solve.

        temp_folder = os.path.join(ModelStore.root_path, str(uuid.uuid4().hex))
        os.mkdir(temp_folder)

        # download each file
        print("Downloading model {} ...".format(model_name))
        current_part = 0
        for current_part in range(parts):
            current_file = os.path.join(temp_folder, "{}.{}".format(model_name, current_part))
            current_url = "{}.{}".format(model_url, current_part)
            status_code = ModelStore.__download_file(current_url, current_file, description=f"    ... download model part {current_part+1}/{parts}")
            if status_code != 200:
                raise Exception(f"Error downloading file {current_url}, received status code {status_code}")

        print("Merging model parts ...")
        zip_file = os.path.join(temp_folder, "archive.zip")
        with open(zip_file, "wb") as f:
            for i in range(parts):
                with open(os.path.join(temp_folder, "{}.{}".format(model_name, i)), "rb") as r:
                    f.write(r.read())

        print("Unzipping ...")
        os.makedirs(model_folder, exist_ok=True)
        zip = zipfile.ZipFile(zip_file, "r")
        zip.extractall(model_folder)
        zip.close()

        print("Cleaning up ...")
        if os.path.exists(temp_folder) and os.path.isdir(temp_folder):
            shutil.rmtree(temp_folder)

        # todo sanity check
        print("Model downloaded successfully!")

        return os.path.abspath(model_folder)

    @staticmethod
    def __download_file(url: str, filename: str, description=None):
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise Exception(f"Error getting {url}, received status_code {r.status_code}")
        file_size = int(r.headers['Content-Length'])
        chunk_size = 1024

        with open(filename, 'wb') as fp:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=description, unit_divisor=1024, disable= True if description is None else False, leave = False) as progressbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk is not None:
                        fp.write(chunk)
                        fp.flush()
                        progressbar.update(len(chunk))

        return r.status_code

    @staticmethod
    def _pack_model(input_folder: str, output_folder: str, model_name:str, split_size_in_mb: int = 50) -> int:
        """
        Zips everything in input folder, splits it in output_folder with model_name.<part##>, return number of parts
        """

        # zip all files in input_folder as input_folder/archive.zip
        print(f"Zipping files from {input_folder}:")
        zip_file_path = os.path.join(input_folder, "archive.zip")
        zip = zipfile.ZipFile(zip_file_path, 'w', compression=zipfile.ZIP_DEFLATED)
        root_len = len(os.path.abspath(input_folder))
        for root, dirs, files in os.walk(input_folder):
            archive_root = os.path.abspath(root)[root_len:]
            for f in files:
                if ".zip" in f:
                    continue
                fullpath = os.path.join(root, f)
                archive_name = os.path.join(archive_root, f)
                print(f"\t adding {fullpath} ...")
                zip.write(fullpath, archive_name, zipfile.ZIP_DEFLATED)

        zip.close()

        # split archive.zip in shards in output_folder
        counter = 0
        with open(zip_file_path, "rb") as f:
            while True:
                byte_s = f.read(split_size_in_mb * 1024 * 1024)
                if not byte_s:
                    break
                chunk_file_path = os.path.join(output_folder, "{}.{}".format(model_name, counter))
                logging.info("\t writing {:.2f}MB to {} ...".format(len(byte_s) / 1024 / 1024, chunk_file_path))
                with open(chunk_file_path, "wb") as r:
                    r.write(byte_s)
                counter += 1

        # return number of files
        return counter




# ensure we get a valid root path for local model storage
ModelStore.root_path = os.path.join(str(Path.home()), ".nlpcube")
if not os.path.exists(ModelStore.root_path):
    os.makedirs(ModelStore.root_path, exist_ok=True)
