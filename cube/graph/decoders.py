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

from collections import namedtuple
import numpy as np
import sys

Arc = namedtuple('Arc', ('tail', 'weight', 'head'))


class GreedyDecoder:
    def _valid(self, arc, tree):
        # just one head
        for sa in tree:
            if sa.tail == arc.tail:
                return False
        stack = [arc.head]
        pos = 0
        used = [False] * len(tree)
        while pos < len(stack):
            for zz in range(len(tree)):
                if tree[zz].tail == stack[pos] and not used[zz]:
                    used[zz] = True
                    stack.append(tree[zz].head)
                    if tree[zz].head == arc.tail:
                        return False
            pos += 1
            # print pos,len(stack)
        return True

    def _get_sort_key(self, item):
        return item.weight

    def _greedy_tree(self, arcs):
        arcs = sorted(arcs, key=self._get_sort_key, reverse=True)
        # print arcs
        final_tree = []
        for index in range(len(arcs)):
            if self._valid(arcs[index], final_tree):
                final_tree.append(arcs[index])
                # print arcs[index]
        return final_tree

    def _make_ordered_list(self, tree, nWords):
        lst = [0] * nWords  # np.zeros(nWords)
        for arc in tree:
            # arc = tree[index]
            tail = arc.tail
            head = arc.head
            lst[tail] = head
        return lst[1:]

    def decode(self, norm_score):
        nWords = len(norm_score)
        g = []
        for iSrc in range(1, nWords):
            for iDst in range(1, nWords):
                if iDst != iSrc:
                    a = Arc(iSrc, norm_score[iSrc][iDst].value(), iDst)
                    g.append(a)
        tree = self._greedy_tree(g)
        best_tree = self._make_ordered_list(tree, nWords)
        return best_tree


