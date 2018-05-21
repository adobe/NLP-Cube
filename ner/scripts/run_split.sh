#!/bin/sh
export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread$
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/

python2 tagger/main.py --split corpus/BG/train.cupt corpus/BG/dev.cupt corpus/BG/resplit
python2 tagger/main.py --split corpus/DE/train.cupt corpus/DE/dev.cupt corpus/DE/resplit
python2 tagger/main.py --split corpus/EL/train.cupt corpus/EL/dev.cupt corpus/EL/resplit
python2 tagger/main.py --split corpus/EN/train.cupt corpus/EN/dev.cupt corpus/EN/resplit
python2 tagger/main.py --split corpus/ES/train.cupt corpus/ES/dev.cupt corpus/ES/resplit
python2 tagger/main.py --split corpus/EU/train.cupt corpus/EU/dev.cupt corpus/EU/resplit
python2 tagger/main.py --split corpus/FA/train.cupt corpus/FA/dev.cupt corpus/FA/resplit
python2 tagger/main.py --split corpus/FR/train.cupt corpus/FR/dev.cupt corpus/FR/resplit
python2 tagger/main.py --split corpus/HE/train.cupt corpus/HE/dev.cupt corpus/HE/resplit

python2 tagger/main.py --split corpus/HI/train.cupt corpus/HI/dev.cupt corpus/HI/resplit
python2 tagger/main.py --split corpus/HR/train.cupt corpus/HR/dev.cupt corpus/HR/resplit
python2 tagger/main.py --split corpus/HU/train.cupt corpus/HU/dev.cupt corpus/HU/resplit
python2 tagger/main.py --split corpus/IT/train.cupt corpus/IT/dev.cupt corpus/IT/resplit
python2 tagger/main.py --split corpus/LT/train.cupt corpus/LT/dev.cupt corpus/LT/resplit
python2 tagger/main.py --split corpus/PL/train.cupt corpus/PL/dev.cupt corpus/PL/resplit
python2 tagger/main.py --split corpus/PT/train.cupt corpus/PT/dev.cupt corpus/PT/resplit
python2 tagger/main.py --split corpus/RO/train.cupt corpus/RO/dev.cupt corpus/RO/resplit
python2 tagger/main.py --split corpus/SL/train.cupt corpus/SL/dev.cupt corpus/SL/resplit
python2 tagger/main.py --split corpus/TR/train.cupt corpus/TR/dev.cupt corpus/TR/resplit
