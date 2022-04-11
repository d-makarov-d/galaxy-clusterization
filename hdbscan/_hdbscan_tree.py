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


import numpy as np


class TreeUnionFind:
    def __init__(self, size):
        self._data = np.zeros((size, 2), dtype=np.int_)
        self._data.T[0] = np.arange(size)
        self.is_component = np.ones(size, dtype=bool)

    def union_(self, x: np.int_, y: np.int_):
        x_root = self.find(x)
        y_root = self.find(y)

        if self._data[x_root, 1] < self._data[y_root, 1]:
            self._data[x_root, 0] = y_root
        elif self._data[x_root, 1] > self._data[y_root, 1]:
            self._data[y_root, 0] = x_root
        else:
            self._data[y_root, 0] = x_root
            self._data[x_root, 1] += 1

    def find(self, x: np.int_):
        if self._data[x, 0] != x:
            self._data[x, 0] = self.find(self._data[x, 0])
            self.is_component[x] = False
        return self._data[x, 0]

    def components(self):
        return self.is_component.nonzero()[0]


def _bfs_from_hierarchy(hierarchy: np.ndarray, bfs_root: np.int_) -> list[np.float_]:
    """
    Perform a breadth first search on a tree in scipy hclust format.
    """
    dim = hierarchy.shape[0]
    max_node = 2 * dim
    num_points = max_node - dim + 1

    to_process = [bfs_root]
    result = []

    while to_process:
        result.extend(to_process)
        to_process = [x - num_points for x in to_process if x >= num_points]
        if to_process:
            to_process = hierarchy[to_process, :2].flatten().astype(np.int_).tolist()

    return result


def _bfs_from_cluster_tree(tree: np.ndarray, bfs_root: np.int_):
    result = []
    to_process = np.array([bfs_root], dtype=np.int_)

    while to_process.shape[0] > 0:
        result.extend(to_process.tolist())
        to_process = tree['child'][np.in1d(tree['parent'], to_process)]

    return result


def _traverse_upwards(
        cluster_tree: np.ndarray,
        cluster_selection_epsilon: np.float_,
        leaf: np.int_,
        allow_single_cluster: np.int_
) -> np.int_:
    root = cluster_tree['parent'].min()
    parent = cluster_tree[cluster_tree['child'] == leaf]['parent'][0]
    if parent == root:
        if allow_single_cluster:
            return parent
        else:
            return leaf  # return node closest to root

    parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['lambda_val']
    if parent_eps > cluster_selection_epsilon:
        return parent
    else:
        return _traverse_upwards(cluster_tree, cluster_selection_epsilon, parent, allow_single_cluster)


def _epsilon_search(
        leaves: set,
        cluster_tree: np.ndarray,
        cluster_selection_epsilon: np.float_,
        allow_single_cluster: np.int_
):
    selected_clusters = list()
    processed = list()

    for leaf in leaves:
        eps = 1 / cluster_tree['lambda_val'][cluster_tree['child'] == leaf][0]
        if eps < cluster_selection_epsilon:
            if leaf not in processed:
                epsilon_child = _traverse_upwards(cluster_tree, cluster_selection_epsilon, leaf, allow_single_cluster)
                selected_clusters.append(epsilon_child)

                for sub_node in _bfs_from_cluster_tree(cluster_tree, epsilon_child):
                    if sub_node != epsilon_child:
                        processed.append(sub_node)
        else:
            selected_clusters.append(leaf)

    return set(selected_clusters)


def _do_labelling(
        tree: np.ndarray,
        clusters: set,
        cluster_label_map: dict,
        allow_single_cluster: np.int_,
        cluster_selection_epsilon: np.float_
) -> np.ndarray:
    child_array = tree['child']
    parent_array = tree['parent']
    lambda_array = tree['lambda_val']

    root_cluster = parent_array.min()
    result = np.empty(root_cluster, dtype=np.int_)

    union_find = TreeUnionFind(parent_array.max() + 1)

    for n in range(tree.shape[0]):
        child = child_array[n]
        parent = parent_array[n]
        if child not in clusters:
            union_find.union_(parent, child)

    for n in range(root_cluster):
        cluster = union_find.find(n)
        if cluster < root_cluster:
            result[n] = -1
        elif cluster == root_cluster:
            if len(clusters) == 1 and allow_single_cluster:
                # TODO check dictionary indexing
                if cluster_selection_epsilon != 0.0:
                    if tree['lambda_val'][tree['child'] == n] >= 1 / cluster_selection_epsilon:
                        result[n] = cluster_label_map[cluster]
                    else:
                        result[n] = -1
                elif tree['lambda_val'][tree['child'] == n] >= \
                        tree['lambda_val'][tree['parent'] == cluster].max():
                    result[n] = cluster_label_map[cluster]
                else:
                    result[n] = -1
            else:
                result[n] = -1
        else:
            result[n] = cluster_label_map[cluster]

    return result


def _recurse_leaf_dfs(cluster_tree: np.ndarray, current_node: np.int_):
    children = cluster_tree[cluster_tree['parent'] == current_node]['child']
    if len(children) == 0:
        return [current_node, ]
    else:
        return sum([_recurse_leaf_dfs(cluster_tree, child) for child in children], [])


def _get_cluster_tree_leaves(cluster_tree: np.ndarray):
    if cluster_tree.shape[0] == 0:
        return []
    root = cluster_tree['parent'].min()
    return _recurse_leaf_dfs(cluster_tree, root)


def _max_lambdas(tree: np.ndarray):
    largest_parent = tree['parent'].max()

    sorted_parent_data = np.sort(tree[['parent', 'lambda_val']], axis=0)
    deaths_arr = np.zeros(largest_parent + 1, dtype=np.float_)
    deaths = deaths_arr
    sorted_parents = sorted_parent_data['parent']
    sorted_lambdas = sorted_parent_data['lambda_val']

    current_parent = -1
    max_lambda = 0

    for row in range(sorted_parent_data.shape[0]):
        parent = sorted_parents[row]
        lambda_ = sorted_lambdas[row]

        if parent == current_parent:
            max_lambda = max(max_lambda, lambda_)
        elif current_parent != -1:
            deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_
        else:
            # Initialize
            current_parent = parent
            max_lambda = lambda_

    deaths[current_parent] = max_lambda  # value for last parent

    return deaths_arr


def _get_probabilities(tree: np.ndarray, cluster_map: dict, labels: np.ndarray):
    child_array = tree['child']
    parent_array = tree['parent']
    lambda_array = tree['lambda_val']

    result = np.zeros(labels.shape[0])
    deaths = _max_lambdas(tree)
    root_cluster = parent_array.min()

    for n in range(tree.shape[0]):
        point = child_array[n]
        if point >= root_cluster:
            continue

        cluster_num = labels[point]

        if cluster_num == -1:
            continue

        cluster = np.int_(cluster_map[cluster_num])
        max_lambda = deaths[cluster]
        if max_lambda == 0.0 or not np.isfinite(lambda_array[n]):
            result[point] = 1.0
        else:
            lambda_ = min(lambda_array[n], max_lambda)
            result[point] = lambda_ / max_lambda

    return result


def _get_stability_scores(labels: np.ndarray, clusters: set, stability: dict, max_lambda: np.float_):
    result = np.empty(len(clusters), dtype=np.float_)
    for n, c in enumerate(sorted(list(clusters))):
        cluster_size = np.sum(labels == n)
        if np.isinf(max_lambda) or max_lambda == 0.0 or cluster_size == 0:
            result[n] = 1.0
        else:
            result[n] = stability[c] / (cluster_size * max_lambda)

    return result


def condense_tree(hierarchy: np.ndarray, min_cluster_size: np.int_) -> np.recarray:
    """Condense a tree according to a minimum cluster size. This is akin
    to the runt pruning procedure of Stuetzle. The result is a much simpler
    tree that is easier to visualize. We include extra information on the
    lambda value at which individual points depart clusters for later
    analysis and computation.

    Parameters
    ----------
    hierarchy : ndarray (n_samples, 4)
        A single linkage hierarchy in scipy.cluster.hierarchy format.

    min_cluster_size : int, optional (default 10)
        The minimum size of clusters to consider. Smaller "runt"
        clusters are pruned from the tree.

    Returns
    -------
    condensed_tree : numpy recarray
        Effectively an edgelist with a parent, child, lambda_val
        and child_size in each row providing a tree structure.
    """

    root = 2 * hierarchy.shape[0]
    num_points = root // 2 + 1
    next_label = num_points + 1

    node_list = _bfs_from_hierarchy(hierarchy, root)

    relabel = np.empty(root + 1, dtype=np.int_)
    relabel[root] = num_points
    result_list = []
    ignore = np.zeros(len(node_list), dtype=np.int_)

    for node in node_list:
        if ignore[node] or node < num_points:
            continue

        children = hierarchy[node - num_points]
        left = np.int_(children[0])
        right = np.int_(children[1])
        if children[2] > 0.0:
            lambda_value = 1.0 / children[2]
        else:
            lambda_value = np.inf

        if left >= num_points:
            left_count = hierarchy[left - num_points][3]
        else:
            left_count = 1

        if right >= num_points:
            right_count = hierarchy[right - num_points][3]
        else:
            right_count = 1

        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            relabel[left] = next_label
            next_label += 1
            result_list.append((relabel[node], relabel[left], lambda_value, left_count))

            relabel[right] = next_label
            next_label += 1
            result_list.append((relabel[node], relabel[right], lambda_value, right_count))

        elif left_count < min_cluster_size and right_count < min_cluster_size:
            for sub_node in _bfs_from_hierarchy(hierarchy, left):
                if sub_node < num_points:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                ignore[sub_node] = True

            for sub_node in _bfs_from_hierarchy(hierarchy, right):
                if sub_node < num_points:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                ignore[sub_node] = True

        elif left_count < min_cluster_size:
            relabel[right] = relabel[node]
            for sub_node in _bfs_from_hierarchy(hierarchy, left):
                if sub_node < num_points:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                ignore[sub_node] = True

        else:
            relabel[left] = relabel[node]
            for sub_node in _bfs_from_hierarchy(hierarchy, right):
                if sub_node < num_points:
                    result_list.append((relabel[node], sub_node, lambda_value, 1))
                ignore[sub_node] = True

    return np.array(result_list, dtype=[('parent', np.int_),
                                        ('child', np.int_),
                                        ('lambda_val', np.float_),
                                        ('child_size', np.int_)])


def compute_stability(condensed_tree: np.ndarray) -> dict:
    """TODO: comments"""
    largest_child = condensed_tree['child'].max()
    smallest_cluster = condensed_tree['parent'].min()
    num_clusters = condensed_tree['parent'].max() - smallest_cluster + 1

    if largest_child < smallest_cluster:
        largest_child = smallest_cluster

    sorted_child_data = np.sort(condensed_tree[['child', 'lambda_val']], axis=0)
    births_arr = np.full(largest_child + 1, np.nan, dtype=np.float_)
    births = births_arr
    sorted_children = sorted_child_data['child'].copy()
    sorted_lambdas = sorted_child_data['lambda_val'].copy()

    parents = condensed_tree['parent']
    sizes = condensed_tree['child_size']
    lambdas = condensed_tree['lambda_val']

    current_child = -1
    min_lambda = 0

    for row in range(sorted_child_data.shape[0]):
        child = sorted_children[row]
        lambda_ = sorted_lambdas[row]

        if child == current_child:
            min_lambda = min(min_lambda, lambda_)
        elif current_child != -1:
            births[current_child] = min_lambda
            current_child = child
            min_lambda = lambda_
        else:
            # Initialize
            current_child = child
            min_lambda = lambda_

    if current_child != -1:
        births[current_child] = min_lambda
    births[smallest_cluster] = 0.0

    result_arr = np.zeros(num_clusters, dtype=np.float_)

    for i in range(condensed_tree.shape[0]):
        parent = parents[i]
        lambda_ = lambdas[i]
        child_size = sizes[i]
        result_index = parent - smallest_cluster

        result_arr[result_index] += (lambda_ - births[parent]) * child_size

    result_pre_dict = np.vstack((np.arange(smallest_cluster, condensed_tree['parent'].max() + 1), result_arr)).T

    return dict(result_pre_dict)


def get_clusters(
        tree: np.ndarray,
        stability: dict,
        cluster_selection_method='eom',
        allow_single_cluster=False,
        cluster_selection_epsilon=0.0,
        max_cluster_size=0
):
    """Given a tree and stability dict, produce the cluster labels
    (and probabilities) for a flat clustering based on the chosen
    cluster selection method.

    Parameters
    ----------
    tree : numpy recarray
        The condensed tree to extract flat clusters from

    stability : dict
        A dictionary mapping cluster_ids to stability values

    cluster_selection_method : string, optional (default 'eom')
        The method of selecting clusters. The default is the
        Excess of Mass algorithm specified by 'eom'. The alternate
        option is 'leaf'.

    allow_single_cluster : boolean, optional (default False)
        Whether to allow a single cluster to be selected by the
        Excess of Mass algorithm.

    cluster_selection_epsilon: float, optional (default 0.0)
        A distance threshold for cluster splits.

    max_cluster_size: int, optional (default 0)
        The maximum size for clusters located by the EOM clusterer. Can
        be overridden by the cluster_selection_epsilon parameter in
        rare cases.

    Returns
    -------
    labels : ndarray (n_samples,)
        An integer array of cluster labels, with -1 denoting noise.

    probabilities : ndarray (n_samples,)
        The cluster membership strength of each sample.

    stabilities : ndarray (n_clusters,)
        The cluster coherence strengths of each cluster.
    """
    # Assume clusters are ordered by numeric id equivalent to
    # a topological sort of the tree; This is valid given the
    # current implementation above, so don't change that ... or
    # if you do, change this accordingly!
    if allow_single_cluster:
        node_list = sorted(stability.keys(), reverse=True)
    else:
        node_list = sorted(stability.keys(), reverse=True)[:-1]
        # (exclude root)

    cluster_tree = tree[tree['child_size'] > 1]
    is_cluster = {cluster: True for cluster in node_list}
    num_points = np.max(tree[tree['child_size'] == 1]['child']) + 1
    max_lambda = np.max(tree['lambda_val'])

    if max_cluster_size <= 0:
        max_cluster_size = num_points + 1  # Set to a value that will never be triggered
    cluster_sizes = {child: child_size for child, child_size
                     in zip(cluster_tree['child'], cluster_tree['child_size'])}
    if allow_single_cluster:
        # Compute cluster size for the root node
        cluster_sizes[node_list[-1]] = np.sum(cluster_tree[cluster_tree['parent'] == node_list[-1]]['child_size'])

    if cluster_selection_method == 'eom':
        for node in node_list:
            child_selection = cluster_tree['parent'] == node
            subtree_stability = np.sum([
                stability[child] for
                child in cluster_tree['child'][child_selection]
            ])
            if subtree_stability > stability[node] or cluster_sizes[node] > max_cluster_size:
                is_cluster[node] = False
                stability[node] = subtree_stability
            else:
                for sub_node in _bfs_from_cluster_tree(cluster_tree, node):
                    if sub_node != node:
                        is_cluster[sub_node] = False

        if cluster_selection_epsilon != 0.0 and cluster_tree.shape[0] > 0:
            eom_clusters = [c for c in is_cluster if is_cluster[c]]
            selected_clusters = []
            # first check if eom_clusters only has root node, which skips epsilon check.
            if len(eom_clusters) == 1 and eom_clusters[0] == cluster_tree['parent'].min():
                if allow_single_cluster:
                    selected_clusters = eom_clusters
            else:
                selected_clusters = _epsilon_search(
                    set(eom_clusters), cluster_tree, cluster_selection_epsilon, allow_single_cluster
                )
            for c in is_cluster:
                if c in selected_clusters:
                    is_cluster[c] = True
                else:
                    is_cluster[c] = False

    elif cluster_selection_method == 'leaf':
        leaves = set(_get_cluster_tree_leaves(cluster_tree))
        if len(leaves) == 0:
            for c in is_cluster:
                is_cluster[c] = False
            is_cluster[tree['parent'].min()] = True

        if cluster_selection_epsilon != 0.0:
            selected_clusters = _epsilon_search(leaves, cluster_tree, cluster_selection_epsilon, allow_single_cluster)
        else:
            selected_clusters = leaves

        for c in is_cluster:
            if c in selected_clusters:
                is_cluster[c] = True
            else:
                is_cluster[c] = False
    else:
        raise ValueError('Invalid Cluster Selection Method: %s\n'
                         'Should be one of: "eom", "leaf"\n')

    clusters = set([c for c in is_cluster if is_cluster[c]])
    cluster_map = {c: n for n, c in enumerate(sorted(list(clusters)))}
    reverse_cluster_map = {n: c for c, n in cluster_map.items()}

    labels = _do_labelling(tree, clusters, cluster_map, allow_single_cluster, cluster_selection_epsilon)
    probs = _get_probabilities(tree, reverse_cluster_map, labels)
    stabilities = _get_stability_scores(labels, clusters, stability, max_lambda)

    return labels, probs, stabilities
