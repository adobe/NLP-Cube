#!/bin/sh
echo "Evaluating"

echo "python2 ~/Downloads/sharedtask-data-master/1.1/bin/evaluate.py --train corpus/$1/train.cupt --gold corpus/$1/test.cupt --pred corpus/test/$1/test.system.cupt"
~/Downloads/sharedtask-data-master/1.1/bin/evaluate.py --train corpus/$1/train.cupt --gold corpus/$1/test.cupt --pred corpus/test/$1/test.system.cupt
