#
# Author: Tiberiu Boros
#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import os
import time

def fopen (filename, mode="r"):
    if sys.version_info[0] == 2:                    
        return open(filename, mode)
    else:                    
        if "b" in mode.lower():
            return open(filename, mode)
        else:
            return open(filename, mode, encoding="utf-8")

# return ETA in seconds
def get_eta(progress, total, time_delta, granularity=2):
    percent_done = float(progress) / float(total)
    return "-" if percent_done <= 0 else pretty_time(seconds=float(time_delta) * (1. / percent_done - percent_done),
                                                     granularity=granularity)


# pretty print time (like: 3d:12h:10m:2s)
def pretty_time(seconds, granularity=2):
    intervals = (('w', 604800), ('d', 86400), ('h', 3600), ('m', 60), ('s', 1))
    result = []
    seconds = int(seconds)
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            result.append("{}{}".format(value, name))
    return ':'.join(result[:granularity])


def log_progress(output_base, task_name, best_epoch, best_training_acc=0., best_dev_acc=0., other=None):
    filename = output_base + ".log"
    with fopen(filename, "w") as f:
        if other != None:
            if isinstance(other, list):
                for i in range(len(other)):
                    f.write(other[i] + "\n")
        f.write("\n")
        f.write("task_name=" + task_name + "\n")
        f.write("best_epoch=" + str(best_epoch) + "\n")
        f.write("best_training_acc=" + str(best_training_acc) + "\n")
        f.write("best_dev_acc=" + str(best_dev_acc) + "\n")
        """if other != None:
            for key in other:
                f.write(key+"="+str(other["key"])+"\n")"""


def line_count(filename):
    f = fopen(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read  # loop optimization
    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)
    return lines
