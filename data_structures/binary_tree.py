from __future__ import annotations

from typing import Iterable, Callable
import numpy as np
from abc import ABC, abstractmethod

from db.galaxy import Galaxy
from algorithms import nts_element


def find_node_split_dim(data, node_indices):
    """Find the most sparse dimension to split space"""
    return np.argmax(data[node_indices].max(0) - data[node_indices].min(0))


class NodeData:
    def __init__(self, idx_start: np.int_, idx_end: np.int_, is_leaf: bool, radius: np.float_):
        # TODO comments
        self.idx_start: np.int_ = idx_start
        self.idx_end: np.int_ = idx_end
        self.is_leaf: bool = is_leaf
        self.radius: np.float_ = radius


class NeighboursHeap:
    """Max-heap structure to keep track of neighbour distances"""


class BinaryTree(ABC):
    """Balanced binary tree, used for KD Tree and Ball Tree implementations"""
    def __init__(self,
                 galaxies: Iterable[Galaxy],
                 metric: Callable[[Galaxy, Galaxy], float],
                 leaf_size=40,
                 sample_weight=None,
                 ):
        """
        :param galaxies: Galaxies to be put in tree
        :param metric: Function to calculate the distance between galaxies
        :param leaf_size: Number of points at which to switch to brute-force
        :param sample_weight: TODO
        """
        self.data = np.array(tuple(map(lambda el: el.split_coordinates, galaxies)), dtype=np.float_)
        self._metric = metric
        self._leaf_size = leaf_size
        self._sample_weight = sample_weight

        self.node_bounds = np.empty((1, 1, 1), dtype=np.float_)

        n_samples = self.data.shape[0]

        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points between leaf_size and 2 * leaf_size
        self.n_levels = int(np.log2(max(1., (n_samples - 1) / self._leaf_size)) + 1)
        self.n_nodes = (2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(n_samples, dtype=np.int_)
        self.node_data = np.zeros(self.n_nodes, dtype=NodeData)

        self._update_sample_weight(n_samples, sample_weight)

        # Allocate tree-specific data
        self._recursive_build(0, 0, n_samples)

    @abstractmethod
    def allocate_data(self):
        pass

    @abstractmethod
    def init_node(self: BinaryTree, i_node: np.int_, idx_start: np.int_, idx_end: np.int_):
        """Initialize the node for the dataset stored in tree.data"""
        pass

    def _update_sample_weight(self, n_samples, sample_weight):
        if sample_weight is not None:
            self.sample_weight_arr = np.asarray(
                sample_weight, dtype=np.float_, order='C')
            self.sample_weight = self.sample_weight_arr
            self.sum_weight = np.sum(self.sample_weight)
        else:
            self.sample_weight = None
            self.sample_weight_arr = np.empty(1, dtype=np.float_, order='C')
            self.sum_weight = n_samples

    def _recursive_build(self, i_node: np.int_, idx_start: np.int_, idx_end: np.int_):
        """
        Recursively build the tree
        :param i_node: Node for the current step
        :param idx_start, idx_end: Bounding indices in the idx_array which define the points that
            belong to this node. TODO: mb restructure
        """
        n_points = idx_end - idx_start
        n_mid = int(n_points / 2)

        self.init_node(i_node, idx_start, idx_end)

        if 2 * i_node + 1 >= self.n_nodes:
            self.node_data[i_node].is_leaf = True
        else:
            # split node and recursively construct child nodes.
            self.node_data[i_node].is_leaf = False
            i_max = find_node_split_dim(self.data, self.idx_array[idx_start:idx_end, :])
            nts_element(self.data[:, i_max], self.idx_array, idx_start, idx_end)
            self._recursive_build(2 * i_node + 1, idx_start, idx_start + n_mid)
            self._recursive_build(2 * i_node + 2, idx_start + n_mid, idx_end)
