import json, os, yaml, uuid
import ntpath
from shutil import rmtree, copyfile
import logging
from tqdm.autonotebook import tqdm as tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parent_dir)

from cube.io_utils.modelstore import ModelStore

if __name__ == "__main__":
    # set vars here
    VERSION = "1.0"
    FOLDER_WITH_YAMLS = os.path.abspath("scripts//train//2.7//language//")
    FOLDER_WITH_TRAINED_MODELS = "/media/echo/5CA436CBA436A802/work/models"  # os.path.abspath("models")
    FOLDER_WHERE_TO_OUTPUT_EVERYTHING = "/media/echo/5CA436CBA436A802/work/nlp-cube-models"  # os.path.abspath("nlp-cube-models")
    URL_ROOT_FOR_MODELS = "https://raw.githubusercontent.com/adobe/NLP-Cube-Models/3.0/models/" # !! make sure it ends with a /

    """    
    0. Open existing catalog, create new key "version" if it does not exist
    
    1. Load all yaml files:
        lang_code_id : "lang_code" -> index in original yaml file
        lang_map : "name" -> "lang_code"        
    
    2. For each lang_code, check to see if files are available to form a model. If not, report errors.
    
        3. Copy all files in temp folder
    
        4. Pack all files and split in shards
    
        5. Move in final model dir with name lang_code.version.###
    
        6. For all entries in lang_map, if lang_code is the current one, create entry in catalog if does not exist
            "name":["version","link","lang_code_index"] 
    
    
    7. Write catalog 
    """

    logger = logging.getLogger("cube")
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(
        fmt="[%(levelname)8s | %(asctime)s | %(filename)-20s:%(lineno)3s | %(funcName)-26s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)



    # STEP 0 - open current catalog or create new
    catalog_path = os.path.join(FOLDER_WHERE_TO_OUTPUT_EVERYTHING, "catalog.json")
    if os.path.exists(catalog_path):
        catalog = json.load(open(catalog_path, "r", encoding="utf8"))
    else:
        catalog = {}


    # STEP 1 - load all yamls
    langcode_to_index = {}
    name_to_langcode = {}

    yamls = os.listdir(FOLDER_WITH_YAMLS) # this just lists the files without path
    yamls = [os.path.abspath(os.path.join(FOLDER_WITH_YAMLS, x)) for x in yamls] # fill full path in
    yamls = [x for x in yamls if os.path.isfile(x) and x.endswith(".yaml")] # filter out possible junk
    print("I see {} yamls in {}.".format(len(yamls), FOLDER_WITH_YAMLS))

    for yaml_file in yamls:
        y = yaml.safe_load(open(yaml_file, "r"))

        for index, lang_code in enumerate(y["language_codes"]):
            if lang_code in langcode_to_index:
                print(f"Warning! lang_code {lang_code} already found in langcode_to_index!")
            langcode_to_index[lang_code] = index

        for name in y["language_map"]:
            if name in name_to_langcode:
                print(f"Warning! name {name} already found in name_to_langcode!")
            name_to_langcode[name] = y["language_map"][name]

    # STEP 2 - for each lang_code
    for lang_code in langcode_to_index:
        print(f"Working on {lang_code} ...")

        # STEP 3 - copy all relevant files for this language_code in temp folder
        files = os.listdir(FOLDER_WITH_TRAINED_MODELS)  # this just lists the files without path
        files = [os.path.abspath(os.path.join(FOLDER_WITH_TRAINED_MODELS, x)) for x in files]  # fill full path in
        files = [x for x in files if os.path.isfile(x) and ".last" not in x]

        # get major language_code
        tt = [x for x in files if lang_code in x]
        if len(tt) == 0:
            print(f"\tCould not get major language code!")
            continue
        tt = tt[0]
        tt = ntpath.basename(tt)
        major_lang_code = tt.split("-")[0]
        print(f"\t major language code is [{major_lang_code}]")

        valid_files = []
        # copy encodings and config
        for f in files:
            ff = ntpath.basename(f)
            if ff.startswith(major_lang_code+"-") and (".config" in ff or ".encodings" in ff):
                valid_files.append(f)
        # copy lang_codes with tok, best and las
        for f in files:
            if lang_code in f and (".tok" in f or ".best" in f or ".las" in f):
                valid_files.append(f)
        files = valid_files

        # check they are valid
        found_tokenizer, found_lemmatizer, found_parser = False, False, False
        for f in files:
            if "tokenizer" in f:
                found_tokenizer = True
            if "lemmatizer" in f:
                found_lemmatizer = True
            if "parser" in f:
                found_parser = True

        if not(found_tokenizer and found_lemmatizer and found_parser):
            print(f"\t {lang_code} does not have all files: tokenizer={found_tokenizer}, lemmatizer={found_lemmatizer}, parser={found_parser}, skipping")
            with open("log.txt", "a") as f:
                f.write(f"\t {lang_code} does not have all files: tokenizer={found_tokenizer}, lemmatizer={found_lemmatizer}, parser={found_parser}, skipping\n")
            continue

        temp_folder = os.path.join(FOLDER_WHERE_TO_OUTPUT_EVERYTHING, str(uuid.uuid4().hex))
        os.mkdir(temp_folder)

        # copy files to temp folder
        print("\t copying files to temp folder ... ")
        for src_file in files:
            _, name = os.path.split(src_file)
            dst_file = os.path.join(temp_folder, name)
            # print((src_file, dst_file))
            copyfile(src_file, dst_file)

        # pack folder in zip file
        zip_file_path = os.path.join(temp_folder, lang_code + "-" + VERSION)
        split_count = ModelStore._pack_model(
            input_folder=temp_folder,
            output_folder=FOLDER_WHERE_TO_OUTPUT_EVERYTHING,
            model_name=lang_code + "-" + VERSION,
            split_size_in_mb=99)

        # delete temp folder
        print("\t deleting temp folder ...")
        if os.path.exists(temp_folder):
            rmtree(temp_folder, ignore_errors=True)

        # STEP 7 - make a catalog entry for all language names affected :
        entry = {
            "version": VERSION,
            "link": URL_ROOT_FOR_MODELS + lang_code + "-" + VERSION,
            "langid": langcode_to_index[lang_code],
            "parts": split_count
        }

        for name in name_to_langcode:
            if name_to_langcode[name] == lang_code:
                print(f"\t making a catalog entry for [{name}] -> [{lang_code}], {split_count} parts, langid {langcode_to_index[lang_code]}")
                if name not in catalog:
                    catalog[name] = []

                catalog[name].append(entry)

    print("Finished processing all language codes, writing catalog ... ")

    json.dump(catalog, open(catalog_path, "w", encoding="utf8"), indent=4, sort_keys=True)

    print("Done.")
