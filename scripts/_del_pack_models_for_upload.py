""" this needs to be deleted """

import json, os, sys
# Append parent dir to sys path.
from shutil import rmtree, copyfile
import logging
from tqdm.autonotebook import tqdm as tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parent_dir)

from cube.io_utils.modelstore import ModelStore

def check (filename, language_family, files):
    for file in files:
        if filename in file:
            return 1
    print(f"{language_family} is missing [{filename}].\n")
    with open("log.txt", "a") as f:
        f.write(f"{language_family} is missing [{filename}].\n")
    return 0

if __name__ == "__main__":
    refresh = False
    folder_with_base_confs = os.path.abspath("scripts//train//2.7//language//")
    folder_with_all_trained_models = "/media/echo/5CA436CBA436A802/work/models" #os.path.abspath("models")
    folder_where_to_output_everything = "/media/echo/5CA436CBA436A802/work/nlp-cube-models" #os.path.abspath("nlp-cube-models")
    url_root_for_models = "https://github.com/adobe/NLP-Cube-Models/blob/3.0/models/" # !! make sure it ends with an /

    logger = logging.getLogger("cube")
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(
        fmt="[%(levelname)8s | %(asctime)s | %(filename)-20s:%(lineno)3s | %(funcName)-26s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)


    """
        Read base confs and get a list of language families
    """
    language_family_confs = os.listdir(folder_with_base_confs) # this just lists the files without path
    language_family_confs = [os.path.abspath(os.path.join(folder_with_base_confs, x)) for x in language_family_confs]
    language_family_confs = [x for x in language_family_confs if os.path.isfile(x) and x.endswith(".yaml")]
    print("I see {} language families in {}.".format(len(language_family_confs), folder_with_base_confs))

    d = {}
    for f in language_family_confs:
        js = json.load(open(f, "r"))
        # list of lists where [0] is the lang code
        _, name = os.path.split(f)
        if name == "all.json":
            continue
        d[name.replace(".json","")] = set()

        for e in js:
            p = e[0]
            d[name.replace(".json", "")].add(p)
            if "_" in p:
                d[name.replace(".json", "")].add(p.split("_")[0])
        if len(d[name.replace(".json","")]) == 0:
            del d[name.replace(".json","")]
        else:
            print("\tlanguage family {} has {} codes: {}".format(name.replace(".json",""), len(d[name.replace(".json","")]), d[name.replace(".json","")]))

    """    
        For each language family, create a folder with the packed and split model in output folder
        Mark language family as valid 
    """
    import uuid

    catalog = {}
    catalog["default"] = []

    os.makedirs(folder_where_to_output_everything, exist_ok=True)
    for fam in d:
        print("Processing language family {}:".format(fam))

        # copy all files in a temp folder
        model_files = os.listdir(folder_with_all_trained_models)
        #print(model_files)
        model_files = [x for x in model_files if x.startswith(fam+"-") and ".last" not in x and ".zip" not in x]

        # check model is complete
        list_of_files = [
            "tokenizer.config",
            "tokenizer.encodings",
            "tokenizer.yaml",

        ]
        #check("")

        #print(model_files)
        model_files = [os.path.abspath(os.path.join(folder_with_all_trained_models,x)) for x in model_files]
        if len(model_files) == 0:
            print("\t no model files found, skipping ...")
            continue
        print("\t found {} model files, copying to temp folder ...".format(len(model_files)))

        temp_folder = os.path.join(folder_where_to_output_everything, str(uuid.uuid4().hex))
        os.mkdir(temp_folder)
        for src_file in tqdm(model_files):
            _, name = os.path.split(src_file)
            dst_file = os.path.join(temp_folder, name)
            #print((src_file, dst_file))
            copyfile(src_file, dst_file)

        # pack folder in zip file
        print("\t packing model ... ")
        zip_file_path = os.path.join(temp_folder, fam+".zip")
        ModelStore._pack_model(input_folder = temp_folder, output_file_path = zip_file_path)

        # split for upload to github
        print("\t splitting model zip ...")
        output_folder = os.path.join(folder_where_to_output_everything, fam)
        os.makedirs(output_folder, exist_ok=True)
        ModelStore._split_packed_model(file_path = zip_file_path, output_folder = output_folder)

        # delete temp folder
        print("\t deleting temp folder ...")
        if os.path.exists(temp_folder):
            rmtree(temp_folder, ignore_errors=True)

        # make a catalog entry
        entry = {}
        entry["languages"] = list(d[fam])
        entry["model"] = url_root_for_models + fam
        catalog["default"].append(entry)


    """    
        Generate catalog
    """
    print("Writing catalog ...")
    json.dump(catalog, open(os.path.join(folder_where_to_output_everything, "catalog.json"), "w", encoding="utf8"), indent=4, sort_keys=True)

    print("Done.")
