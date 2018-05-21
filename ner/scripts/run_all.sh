#!/bin/sh
export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/

python2 tagger/main.py --train corpus/BG/train.cupt corpus/BG/dev.cupt models/bg/model-full 10
python2 tagger/main.py --train corpus/DE/train.cupt corpus/DE/dev.cupt models/de/model-full 10
python2 tagger/main.py --train corpus/EL/train.cupt corpus/EL/dev.cupt models/el/model-full 10
python2 tagger/main.py --train corpus/EN/resplit-train.cupt corpus/EN/resplit-dev.cupt models/en/model-full 10
python2 tagger/main.py --train corpus/ES/train.cupt corpus/ES/dev.cupt models/es/model-full 10
python2 tagger/main.py --train corpus/EU/train.cupt corpus/EU/dev.cupt models/eu/model-full 10
python2 tagger/main.py --train corpus/FA/train.cupt corpus/FA/dev.cupt models/fa/model-full 10
python2 tagger/main.py --train corpus/FR/train.cupt corpus/FR/dev.cupt models/fr/model-full 10
python2 tagger/main.py --train corpus/HE/train.cupt corpus/HE/dev.cupt models/he/model-full 10

python2 tagger/main.py --train corpus/HI/resplit-train.cupt corpus/HI/resplit-dev.cupt models/hi/model-full 10
python2 tagger/main.py --train corpus/HR/train.cupt corpus/HR/dev.cupt models/hr/model-full 10
python2 tagger/main.py --train corpus/HU/train.cupt corpus/HU/dev.cupt models/hu/model-full 10
python2 tagger/main.py --train corpus/IT/train.cupt corpus/IT/dev.cupt models/it/model-full 10
python2 tagger/main.py --train corpus/LT/resplit-train.cupt corpus/LT/resplit-dev.cupt models/lt/model-full 10
python2 tagger/main.py --train corpus/PL/train.cupt corpus/PL/dev.cupt models/pl/model-full 10
python2 tagger/main.py --train corpus/PT/train.cupt corpus/PT/dev.cupt models/pt/model-full 10
python2 tagger/main.py --train corpus/RO/train.cupt corpus/RO/dev.cupt models/ro/model-full 10
python2 tagger/main.py --train corpus/SL/train.cupt corpus/SL/dev.cupt models/sl/model-full 10
python2 tagger/main.py --train corpus/TR/train.cupt corpus/TR/dev.cupt models/tr/model-full 10
