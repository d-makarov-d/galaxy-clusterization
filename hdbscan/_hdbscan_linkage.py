# Sources in this file contain code from
# https://github.com/scikit-learn-contrib/hdbscan HDBSCAN project licensed
# by the following BSD-3-Clause License
#
# Copyright (c) 2015, Leland McInnes
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Sequence
import numpy as np

from db.galaxy import Galaxy


def mst_linkage_core_vector(raw_data: Sequence[Galaxy], core_distances: np.ndarray,
                            clusterer, alpha=1.0):
    """TODO: comments"""
    n_elements = len(raw_data)

    result = np.zeros((n_elements - 1, 3))
    in_tree = np.zeros(n_elements, dtype=np.int8)
    current_node = 0
    current_distances = np.infty * np.ones(n_elements)

    for i in range(1, n_elements):
        in_tree[current_node] = 1

        current_node_core_distance = core_distances[current_node]

        new_distance = np.inf
        new_node = 0

        for j in range(n_elements):
            if in_tree[j]:
                continue

            right_value = current_distances[j]
            left_value = clusterer.calc_distance(raw_data[current_node], raw_data[j])

            if alpha != 1.0:
                left_value /= alpha

            core_value = core_distances[j]
            if (current_node_core_distance > right_value or
                    core_value > right_value or
                    left_value > right_value):
                if right_value < new_distance:
                    new_distance = right_value
                    new_node = j
                continue

            if core_value > current_node_core_distance:
                if core_value > left_value:
                    left_value = core_value
            else:
                if current_node_core_distance > left_value:
                    left_value = current_node_core_distance

            if left_value < right_value:
                current_distances[j] = left_value
                if left_value < new_distance:
                    new_distance = left_value
                    new_node = j
            else:
                if right_value < new_distance:
                    new_distance = right_value
                    new_node = j

        result[i - 1, 0] = current_node
        result[i - 1, 1] = new_node
        result[i - 1, 2] = new_distance
        current_node = new_node

    return result


class UnionFind:
    def __init__(self, N: np.int_):
        self.parent = np.full(2 * N - 1, -1, dtype=np.int_)
        self.next_label = N
        self.size = np.hstack((np.ones(N, dtype=np.int_), np.zeros(N - 1, dtype=np.int_)))

    def union(self, m: np.int_, n: np.int_):
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

    def fast_find(self, n: np.int_):
        p = n
        while self.parent[n] != -1:
            n = self.parent[n]
        # label up to the root
        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n
        return n


def label(L: np.ndarray):
    """TODO: comments"""
    result = np.zeros((L.shape[0], 4))

    N = L.shape[0] + 1
    U = UnionFind(N)

    for index in range(L.shape[0]):
        a = np.int_(L[index, 0])
        b = np.int_(L[index, 1])
        delta = L[index, 2]

        aa, bb = U.fast_find(a), U.fast_find(b)

        result[index][0] = aa
        result[index][1] = bb
        result[index][2] = delta
        result[index][3] = U.size[aa] + U.size[bb]

        U.union(aa, bb)

    return result
