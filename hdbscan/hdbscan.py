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
from joblib.parallel import cpu_count

from db.galaxy import Galaxy
from data_structures.kd_tree import KDTree
from ._hdbscan_linkage import mst_linkage_core_vector, label
from ._hdbscan_tree import condense_tree, compute_stability, get_clusters
from ._hdbscan_boruvka import KDTreeBoruvkaAlgorithm


def _tree_to_labels(
        single_linkage_tree: np.ndarray,
        min_cluster_size=10,
        cluster_selection_method='eom',
        allow_single_cluster=False,
        cluster_selection_epsilon=0.0,
        max_cluster_size=0
):
    """Converts a pretrained tree and cluster size into a
    set of labels and probabilities.
    """
    condensed_tree = condense_tree(single_linkage_tree,
                                   min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    labels, probabilities, stabilities = get_clusters(
        condensed_tree,
        stability_dict,
        cluster_selection_method,
        allow_single_cluster,
        cluster_selection_epsilon,
        max_cluster_size
    )

    return labels, probabilities, stabilities, condensed_tree, single_linkage_tree


def hdbscan_prims_kdtree(galaxies: Sequence[Galaxy], clusterer, min_samples=5,
                         alpha=1.0, leaf_size=40, gen_min_span_tree=False):
    tree = KDTree(galaxies, clusterer, leaf_size)

    # Get distance to kth nearest neighbour
    core_distances = tree.query(galaxies, k=min_samples, dualtree=True, breadth_first=True)[0][:, -1]

    # Mutual reachability distance is implicit in mst_linkage_core_vector
    min_spanning_tree = mst_linkage_core_vector(galaxies, core_distances, clusterer, alpha)

    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        # TODO: use logger
        print('Cannot generate Minimum Spanning Tree; '
              'the implemented Prim\'s does not produce '
              'the full minimum spanning tree ')

    return single_linkage_tree, None


def hdbscan_boruvka_kdtree(galaxies: Sequence[Galaxy], clusterer, min_samples=5,
                           alpha=1., leaf_size=40, approx_min_span_tree=True,
                           gen_min_span_tree=False, core_dist_n_jobs=4):
    if leaf_size < 3:
        leaf_size = 3

    if core_dist_n_jobs < 1:
        core_dist_n_jobs = max(cpu_count() + 1 + core_dist_n_jobs, 1)

    tree = KDTree(galaxies, clusterer, leaf_size)
    alg = KDTreeBoruvkaAlgorithm(
        tree,
        min_samples,
        leaf_size // 3,
        alpha,
        approx_min_span_tree,
        core_dist_n_jobs
    )

    min_spanning_tree = alg.spanning_tree()
    # Sort edges of the min_spanning_tree by weight
    row_order = np.argsort(min_spanning_tree.T[2])
    min_spanning_tree = min_spanning_tree[row_order, :]
    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None
