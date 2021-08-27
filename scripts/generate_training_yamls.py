import sys, os, yaml
from pprint import pprint
from bs4 import BeautifulSoup

path_to_corpus_folder = "corpus/ud-treebanks-v2.5"
path_to_save_folder = "scripts/train/2.5"

path_to_corpus_folder = "corpus/ud-treebanks-v2.7/"
path_to_save_folder = "scripts/train/2.7"

# read default language map
default_language_map = {}
with open("scripts/default_language.txt","r") as f:
    lines = f.readlines()
    for l in lines:
        p = l.strip().split("\t")
        major, treebank = p[0], p[0]+"_"+p[1]
        default_language_map[major] = treebank

# read UD extracted table file as html <-- copy-paste from UD's website of the table only
with open("scripts/ud_table.html","r", encoding="utf8") as f:
    source = f.read()

soup = BeautifulSoup(source, features="html.parser")
table = []

##### STEP 1, extract data from html table
header = True
for tag in soup.div.findChildren(recursive=False):
    if str(tag).strip() == "" or len(str(tag))<50:
        continue
    if header:
        #print(tag)
        for elem in tag.find_all("span", class_="doublewidespan", recursive=False):
            if elem.text.strip() != "":
                language_name = elem.text
        print("Language name: "+ language_name)

        for elem in tag.find_all("span", class_="triplewidespan", recursive=False):
            if elem.text.strip() != "":
                language_family = [x.strip() for x in elem.text.split(",")]
        print("Language family: {}".format(language_family))

        header = False
    else:
        cheader = True
        for content in tag.find_all("div", class_="ui-accordion-icons", recursive=False): # just 1 content
            for content_div in content.findChildren(recursive=False): # list of divs
                #print(content_div)
                #print("_" * 50)

                if cheader: # get name of treebank
                    for sp in content_div.find_all("span", class_="doublewidespan", recursive=False):
                        #print(sp)
                        if sp.text.strip() != "":
                            treebank_name = sp.text
                    print("Treebank name: "+treebank_name)
                    cheader = False
                else: # get folder name
                    for sp in content_div.find_all("a", href=True):
                        #print(sp["href"])
                        link = str(sp["href"]).strip()
                        if "UniversalDependencies" in link:
                            link = link[41:]
                            treebank_folder =  link[:link.find("/")]
                        elif "treebanks/" in link:
                            link = link[10:]
                            link = link[:link.find("/")]
                            language_code = link
                            major_language_code = link.split("_")[0]

                    print("Treebank folder: " + treebank_folder)
                    print("Language code: " + language_code)
                    print("Major language code: " + major_language_code)

                    if language_name == None:
                        raise Exception("Language name is none")
                    if language_family == None:
                        raise Exception("Language family none")
                    if treebank_name == None:
                        raise Exception("Treebank name is none")
                    if treebank_folder == None:
                        raise Exception("Treebank folder is none")

                    telem = {}
                    telem["language_name"] = language_name
                    telem["language_family"] = language_family
                    telem["treebank_name"] = treebank_name
                    telem["treebank_folder"] = treebank_folder
                    telem["language_code"] = language_code
                    telem["major_language_code"] = major_language_code
                    print(telem)
                    if "simonero" in language_code:
                        continue
                    if "_pud" in language_code:
                        continue
                    table.append(telem)
                    treebank_name, treebank_folder, language_code, major_language_code = None, None, None, None

                    cheader = True

        # reset for next language
        language_name, language_family, treebank_name, treebank_folder = None, None, None, None

        header = True
    #input('\n\n----------\n\n')

##### STEP 2, add paths to train dev test
valid = []
for l in table:
    train_file = os.path.join(path_to_corpus_folder, l["treebank_folder"], l["language_code"] + "-ud-train.conllu")
    dev_file = os.path.join(path_to_corpus_folder, l["treebank_folder"], l["language_code"] + "-ud-dev.conllu")
    test_file = os.path.join(path_to_corpus_folder, l["treebank_folder"], l["language_code"] + "-ud-test.conllu")

    # for PUD treebanks
    if l["language_code"].endswith("pud"): # languages only with test files
        l["train_file"] = [l["language_code"],train_file]
        l["dev_file"] = None
        l["test_file"] = None
    else: # regular treebank
        if os.path.exists(train_file):
            l["train_file"] = [l["language_code"],train_file]
        else:
            l["train_file"] = None
        if os.path.exists(dev_file):
            l["dev_file"] = [l["language_code"],dev_file]
        else:
            l["dev_file"] = None
        if os.path.exists(test_file):
            if l["dev_file"] == None:
                l["dev_file"] = [l["language_code"],test_file]
                l["test_file"] = [l["language_code"],test_file]
            else:
                l["test_file"] = [l["language_code"],test_file]
        else:
            l["test_file"] = None

    if l["train_file"] is not None:
        valid.append(l)
    else:
        print(" Language skipped due to no training file: ")
        pprint(l)
        print("\n")

table = valid


#### STEP 3, generate train files

# generate single language files
print("\n\nRunning single treebank ...\n")
folder = os.path.join(path_to_save_folder, "treebank")
os.makedirs(folder, exist_ok=True)

for l in table:
    obj = {
        "language_map": {l["language_code"]:l["language_code"]},
        "language_codes": [l["language_code"]],
        "train_files": {},
        "dev_files": {},
        "test_files": {}
    }

    if l["train_file"] is not None:
        obj["train_files"][l["train_file"][0]] = l["train_file"][1]
    if l["dev_file"] is not None:
        obj["dev_files"][l["dev_file"][0]] = l["dev_file"][1]
    if l["test_file"] is not None:
        obj["test_files"][l["test_file"][0]] = l["test_file"][1]
    else:
        if l["dev_file"] is not None:
            obj["test_files"][l["dev_file"][0]] = l["dev_file"][1]

    obj["language_codes"] = list(set([obj["language_map"][key] for key in obj["language_map"]]))

    if len(obj["train_files"]) == 0 :
        print("\tLanguage {} has zero training files, skipping. ".format(l["language_code"]))
        continue
    if len(obj["dev_files"]) == 0 :
        print("\tLanguage {} has zero dev files, skipping. ".format(l["language_code"]))
        continue
    if len(obj["test_files"]) == 0:
        print("\tLanguage {} has zero test files, skipping. ".format(l["language_code"]))
        continue

    filename = l["language_code"]+".yaml"
    with open(os.path.join(folder, filename), 'w') as f:
        data = yaml.dump(obj, f, sort_keys=True)

# generate per major language files (e.g. en)
print("\n\nRunning per language treebank aggregation ...\n")
folder = os.path.join(path_to_save_folder, "language")
os.makedirs(folder, exist_ok=True)

groups = {}
for l in table:
    lc = l["major_language_code"]
    if lc not in groups:
        groups[lc] = []
    groups[lc].append(l)

for g in groups:
    if g not in default_language_map:
        #raise Exception("\t\tLanguage {} does not have default language code, skipping!".format(g))
        print("\tLanguage {} ({}) does not have default language code, skipping!".format(groups[g][0]["language_name"], g))
        continue

    obj = {
        "language_map": {},
        "language_codes": [],
        "train_files": {},
        "dev_files": {},
        "test_files": {}
    }
    for l in groups[g]:
        # add full language name, lowercased
        if l["language_name"].lower() not in obj["language_map"]:
            obj["language_map"][l["language_name"].lower()] = default_language_map[g]
        # add major language code
        if g not in obj["language_map"]:
            obj["language_map"][g] = default_language_map[g]
        # add treebank code
        obj["language_map"][l["language_code"]] = l["language_code"]

        if l["train_file"] is not None:
            obj["train_files"][l["train_file"][0]] = l["train_file"][1]
        if l["dev_file"] is not None:
            obj["dev_files"][l["dev_file"][0]] = l["dev_file"][1]
        if l["test_file"] is not None:
            obj["test_files"][l["test_file"][0]] = l["test_file"][1]

    obj["language_codes"] = list(set([obj["language_map"][key] for key in obj["language_map"]]))

    if len(obj["train_files"]) == 0 :
        print("\tLanguage {} ({}) has zero training files, skipping. ".format(l["language_name"], g))
        continue
    if len(obj["dev_files"]) == 0 :
        print("\tLanguage {} ({}) has zero dev files, skipping. ".format(l["language_name"], g))
        continue
    if len(obj["test_files"]) == 0:
        print("\tLanguage {} ({}) has zero test files, skipping. ".format(l["language_name"], g))
        continue

    filename = l["major_language_code"]+".yaml"
    with open(os.path.join(folder, filename), 'w') as f:
        data = yaml.dump(obj, f, sort_keys=True)

# generate per language family
print("\n\nRunning per language family aggregation ...\n")
folder = os.path.join(path_to_save_folder, "family")
os.makedirs(folder, exist_ok=True)

groups = {}
for l in table:
    if l["major_language_code"] not in default_language_map:
        #raise Exception("\t\tLanguage {} does not have default language code, skipping!".format(g))
        print("\tLanguage {} ({}) does not have default language code, skipping!".format(l["language_name"], l["major_language_code"]))
        continue

    families = l["language_family"]
    for f in families:
        if f != "IE" and "Afro-Asiatic" not in f:
            if f not in groups:
                groups[f] = []

            groups[f].append(l)

for g in groups:
    obj = {
        "language_map": {},
        "language_codes": [],
        "train_files": {},
        "dev_files": {},
        "test_files": {}
    }

    for l in groups[g]:
        # add full language name, lowercased
        if l["language_name"].lower() not in obj["language_map"]:
            obj["language_map"][l["language_name"].lower()] = default_language_map[l["major_language_code"]]
        # add major language code
        if g not in obj["language_map"]:
            obj["language_map"][l["major_language_code"]] = default_language_map[l["major_language_code"]]
        # add treebank code
        obj["language_map"][l["language_code"]] = l["language_code"]

        if l["train_file"] is not None:
            obj["train_files"][l["train_file"][0]] = l["train_file"][1]
        if l["dev_file"] is not None:
            obj["dev_files"][l["dev_file"][0]] = l["dev_file"][1]
        if l["test_file"] is not None:
            obj["test_files"][l["test_file"][0]] = l["test_file"][1]

    obj["language_codes"] = list(set([obj["language_map"][key] for key in obj["language_map"]]))

    if len(obj["train_files"]) == 0 :
        print("\tLanguage {} (fam: {}) has zero training files, skipping. ".format(l["language_name"], g))
        pprint(obj)
        continue
    if len(obj["dev_files"]) == 0 :
        print("\tLanguage {} (fam: {}) has zero dev files, skipping. ".format(l["language_name"], g))
        pprint(obj)
        continue
    if len(obj["test_files"]) == 0:
        print("\tLanguage {} (fam: {}) has zero test files, skipping. ".format(l["language_name"], g))
        pprint(obj)
        continue

    filename = g.lower()+".yaml"
    with open(os.path.join(folder, filename), 'w') as f:
        yaml.dump(obj, f, sort_keys=True)
