#!/bin/bash
export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/

python2 tagger/main.py --train corpus/EN/resplit-train.cupt corpus/EN/resplit-dev.cupt models/en/model-full 10
python2 tagger/main.py --train corpus/HI/resplit-train.cupt corpus/HI/resplit-dev.cupt models/hi/model-full 10
python2 tagger/main.py --train corpus/LT/resplit-train.cupt corpus/LT/resplit-dev.cupt models/lt/model-full 10
