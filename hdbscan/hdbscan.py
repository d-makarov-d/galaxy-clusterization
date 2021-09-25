from typing import Sequence
import numpy as np

from db.galaxy import Galaxy
from data_structures.kd_tree import KDTree
from ._hdbscan_linkage import mst_linkage_core_vector, label
from ._hdbscan_tree import condense_tree, compute_stability, get_clusters


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
