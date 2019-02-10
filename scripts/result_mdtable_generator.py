# This script reads the results.json file and converts it to a results.md file

import json, collections

def extract_language_from_test_file(file):
    # ex: "/home/ubuntu/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms/af_afribooms-ud-test.txt"
    parts = file.split("/")
    dir = parts[-2].replace("UD_","")
    dir = dir[:dir.find("-")]
    return dir.replace("_"," ")

all = json.load(open("results.json","r"))

lall = {}
for model, elem in all.items():
    language = extract_language_from_test_file(elem["test_file"])
    print(language)
    if language not in lall:
        lall[language] = {}    
    lall[language][model] = elem
all = collections.OrderedDict(sorted(lall.items()))

header =  "|Language|Model|Token|Sentence|UPOS|XPOS|AllTags|Lemmas|UAS|LAS|\n"
header += "|--------|-----|:---:|:------:|:--:|:--:|:-----:|:----:|:-:|:-:|\n"
rows = []
for language, langdict in all.items():
    lrow = "|"+language+"|\n"
    rows.append(lrow)
    langdict = collections.OrderedDict(sorted(langdict.items()))
    for model, elem in langdict.items():
        row = "| |"
        row += model + "|"        
        row += str(round(elem["metrics"]["Tokens"]["f1"],2)) + "|"
        row += str(round(elem["metrics"]["Sentences"]["f1"],2)) + "|"
        row += str(round(elem["metrics"]["UPOS"]["f1"],2)) + "|"
        row += str(round(elem["metrics"]["XPOS"]["f1"],2)) + "|"
        row += str(round(elem["metrics"]["AllTags"]["f1"],2)) + "|"
        row += str(round(elem["metrics"]["Lemmas"]["f1"],2)) + "|"
        row += str(round(elem["metrics"]["UAS"]["f1"],2)) + "|"
        row += str(round(elem["metrics"]["LAS"]["f1"],2)) + "|\n"
        rows.append(row)
        
with open("results.md","w",encoding="utf8") as f:
    f.write(header)
    for row in rows:
        f.write(row)