# PARSEME - shared task

* [Shared task](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018&subpage=CONF_40_Shared_Task)
* [Shared task - github repo](https://gitlab.com/parseme/sharedtask-data)
* [Format specifications](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018&subpage=CONF_45_Format_specification)

## Data

The shared task covers 20 languages: Arabic (AR), Bulgarian (BG), German (DE), Greek (EL), English (EN), Spanish (ES), Basque (EU), Farsi (FA), French (FR), Hindi (HI), Hebrew (HE), Croatian (HR), Hungarian (HU), Italian (IT), Lithuanian (LT), Polish (PL), Brazilian Portuguese (PT), Romanian (RO), Slovenian (SL), Turkish (TR).

For each language, we provide corpora (in the `.cupt` format) in which VMWEs are annotated according to universal guidelines:

## Running the script


```bash
# Download the data
python2 scripts/download_data.py --language ro
```

# Train the GBD-NER
```bash
$ wget https://gitlab.com/parseme/sharedtask-data/raw/master/1.1/RO/train.cupt -O corpus/ro/train.cupt
$ wget https://gitlab.com/parseme/sharedtask-data/raw/master/1.1/RO/dev.cupt -O corpus/ro/dev.cupt
$ wget https://gitlab.com/parseme/sharedtask-data/raw/master/1.1/RO/test.blind.cupt -O corpus/ro/test.blind.cupt
$ wget https://gitlab.com/parseme/sharedtask-data/raw/master/1.1/RO/test.cupt -O corpus/ro/test.cupt
$ python2 tagger/main.py --train corpus/ro/train.cupt corpus/ro/dev.cupt models/ro/ner 10
```

# Test the system
```
$ python2 tagger/main.py --test models/ro/ner corpus/ro/test.blind.cupt corpus/ro/test.system.cupt
#using the PARSEME official evaluation script:
$ python2 evaluate.py --train corpus/ro/train.cupt --gold corpus/ro/test.cupt --pred corpus/test/ro/test.system.cupt"
```
