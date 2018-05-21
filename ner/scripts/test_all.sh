#!/bin/sh
export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/

python2 tagger/main.py --test models/bg/model-full corpus/test/BG/test.blind.cupt corpus/test/BG/test.system.cupt
python2 tagger/main.py --test models/de/model-full corpus/test/DE/test.blind.cupt corpus/test/DE/test.system.cupt
python2 tagger/main.py --test models/el/model-full corpus/test/EL/test.blind.cupt corpus/test/EL/test.system.cupt
python2 tagger/main.py --test models/en/model-full corpus/test/EN/test.blind.cupt corpus/test/EN/test.system.cupt
python2 tagger/main.py --test models/es/model-full corpus/test/ES/test.blind.cupt corpus/test/ES/test.system.cupt
python2 tagger/main.py --test models/eu/model-full corpus/test/EU/test.blind.cupt corpus/test/EU/test.system.cupt
python2 tagger/main.py --test models/fa/model-full corpus/test/FA/test.blind.cupt corpus/test/FA/test.system.cupt
python2 tagger/main.py --test models/fr/model-full corpus/test/FR/test.blind.cupt corpus/test/FR/test.system.cupt
python2 tagger/main.py --test models/he/model-full corpus/test/HE/test.blind.cupt corpus/test/HE/test.system.cupt
python2 tagger/main.py --test models/hi/model-full corpus/test/HI/test.blind.cupt corpus/test/HI/test.system.cupt
python2 tagger/main.py --test models/hr/model-full corpus/test/HR/test.blind.cupt corpus/test/HR/test.system.cupt
python2 tagger/main.py --test models/hu/model-full corpus/test/HU/test.blind.cupt corpus/test/HU/test.system.cupt
python2 tagger/main.py --test models/it/model-full corpus/test/IT/test.blind.cupt corpus/test/IT/test.system.cupt
python2 tagger/main.py --test models/lt/model-full corpus/test/LT/test.blind.cupt corpus/test/LT/test.system.cupt
python2 tagger/main.py --test models/pl/model-full corpus/test/PL/test.blind.cupt corpus/test/PL/test.system.cupt
python2 tagger/main.py --test models/pt/model-full corpus/test/PT/test.blind.cupt corpus/test/PT/test.system.cupt
python2 tagger/main.py --test models/ro/model-full corpus/test/RO/test.blind.cupt corpus/test/RO/test.system.cupt
python2 tagger/main.py --test models/sl/model-full corpus/test/SL/test.blind.cupt corpus/test/SL/test.system.cupt
python2 tagger/main.py --test models/tr/model-full corpus/test/TR/test.blind.cupt corpus/test/TR/test.system.cupt

