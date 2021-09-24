from __future__ import annotations

import numpy as np

from db.galaxy import Galaxy
from .binary_tree import BinaryTree, NodeData


class KDTree(BinaryTree):
    def allocate_data(self, n_nodes: np.int_, n_features: np.int_):
        self.node_bounds = np.zeros((2, n_nodes, n_features), dtype=np.float_)

    def init_node(self, i_node: np.int_, idx_start: np.int_, idx_end: np.int_):
        """Initialize the node for the dataset stored in self.data"""
        lower_bounds = self.node_bounds[0, i_node, :]
        upper_bounds = self.node_bounds[1, i_node, :]

        # initial Node bounds
        lower_bounds.fill(np.inf)
        upper_bounds.fill(-np.inf)

        # Compute the actual data range.  At build time, this is slightly
        # slower than using the previously-computed bounds of the parent node,
        # but leads to more compact trees and thus faster queries.
        for i in range(idx_start, idx_end):
            data_row = self.data[self.idx_array[i], :]
            lower_bounds[:] = np.minimum(lower_bounds, data_row)
            upper_bounds[:] = np.maximum(upper_bounds, data_row)

        rad = np.sum(np.power(0.5 * np.abs(upper_bounds - lower_bounds), 2))

        # The radius will hold the size of the circumscribed hypersphere: in querying,
        # this is used as a measure of the size of each node when deciding which nodes to split.
        radius = np.sqrt(rad)
        self.node_data[i_node] = NodeData(idx_start, idx_end, is_leaf=False, radius=radius)

    def min_rdist(self, i_node: np.int_, pt: Galaxy) -> np.float_:
        """Compute the minimum reduced-distance between a point and a node TODO: prove that this applicable for astrophysics"""
        d_lo = self.node_bounds[0, i_node, :] - pt.split_coordinates
        d_hi = pt.split_coordinates - self.node_bounds[1, i_node, :]
        d = (d_lo + np.abs(d_lo)) + (d_hi + np.abs(d_hi))
        rdist = np.sum(np.power(0.5 * d, 2))

        return rdist

    @staticmethod
    def min_rdist_dual(tree1: KDTree, i_node1: np.int_, tree2: KDTree, i_node2: np.int_) -> np.float_:
        """Compute the minimum reduced distance between two nodes TODO: prove that this applicable for astrophysics"""
        d1 = tree1.node_bounds[0, i_node1, :] - tree2.node_bounds[1, i_node2, :]
        d2 = tree2.node_bounds[0, i_node2, :] - tree1.node_bounds[1, i_node1, :]
        d = (d1 + np.abs(d1)) + (d2 + np.abs(d2))
        rdist = np.sum(np.power(0.5 * d, 2))

        return rdist
