from __future__ import annotations

from typing import Sequence, Callable
import numpy as np
from abc import ABC, abstractmethod

from db.galaxy import Galaxy
from algorithms import nts_element


def find_node_split_dim(data, node_indices):
    """Find the most sparse dimension to split space"""
    return np.argmax(data[node_indices].max(0) - data[node_indices].min(0))


def _simultaneous_sort(dist, idx):
    """TODO: in c"""
    i = np.argsort(dist)
    return dist[i], idx[i]


class NodeData:
    def __init__(self, idx_start: np.int_, idx_end: np.int_, is_leaf: bool, radius: np.float_):
        # TODO comments
        self.idx_start: np.int_ = idx_start
        self.idx_end: np.int_ = idx_end
        self.is_leaf: bool = is_leaf
        self.radius: np.float_ = radius


class NodeHeapData:
    """TODO: comments"""
    def __init__(self, val: np.float_, i1: np.int_, i2: np.int_):
        self.val: np.float_ = val
        self.i1: np.int_ = i1
        self.i2: np.int_ = i2


class NeighborsHeap:
    """Max-heap structure to keep track of neighbour distances

    This implements an efficient pre-allocated set of fixed-size heaps
    for chasing neighbors, holding both an index and a distance.
    When any row of the heap is full, adding an additional point will push
    the furthest point off the heap."""
    def __init__(self, n_pts: np.int_, n_nbrs: np.int_):
        """ TODO: mb in c
        :param n_pts: Number of heaps to use
        :param n_nbrs: Size of each heap
        """
        self.distances = np.full((n_pts, n_nbrs), np.inf)
        self.indices = np.zeros((n_pts, n_nbrs))

    def get_arrays(self, sort=True) -> tuple[np.ndarray, np.ndarray]:
        """Get the arrays of distances and indices within the heap.

        If sort=True, then simultaneously sort the indices and distances,
        so the closer points are listed first.
        :return Tuple with distances and indices arrays of size (n_pts, n_nbrs)
        """
        if sort:
            self._sort()
        return self.distances, self.indices

    def largest(self, row: np.int_) -> np.float_:
        """Return the largest distance in the given row"""
        return self.distances[row, 0]

    def push(self, row: np.int_, val: np.float_, i_val: np.int_):
        """push (val, i_val) into the given row"""
        size = self.distances.shape[1]
        dist_arr = self.distances[row, :]
        ind_arr = self.indices[row, :]

        # check if val should be in heap
        if val >= dist_arr[0]:
            return

        # insert val at position zero
        dist_arr[0] = val
        ind_arr[0] = i_val

        # descend the heap, swapping values until the max heap criterion is met
        i = 0
        while True:
            ic1 = 2 * i + 1
            ic2 = ic1 + 1

            if ic1 >= size:
                break
            elif ic2 >= size:
                if dist_arr[ic1] > val:
                    i_swap = ic1
                else:
                    break
            elif dist_arr[ic1] >= dist_arr[ic2]:
                if val < dist_arr[ic1]:
                    i_swap = ic1
                else:
                    break
            else:
                if val < dist_arr[ic2]:
                    i_swap = ic2
                else:
                    break

            dist_arr[i] = dist_arr[i_swap]
            ind_arr[i] = ind_arr[i_swap]

            i = i_swap

        dist_arr[i] = val
        ind_arr[i] = i_val

    def _sort(self):
        """simultaneously sort the distances and indices"""
        for row in range(self.distances.shape[0]):
            self.distances[row, :], self.indices[row, :] = \
                _simultaneous_sort(self.distances[row, :], self.indices[row, :])


class NodeHeap:
    """This is a min-heap implementation for keeping track of nodes
       during a breadth-first search.  Unlike the NeighborsHeap above,
       the NodeHeap does not have a fixed size and must be able to grow
       as elements are added.

       Internally, the data is stored in a simple binary heap which meets
       the min heap condition:

           heap[i].val < min(heap[2 * i + 1].val, heap[2 * i + 2].val)
    """
    def __init__(self, size_guess=100):
        size_guess = max(size_guess, 1)  # need space for at least one item
        self.data = np.zeros(size_guess, dtype=NodeHeapData)
        self.n = size_guess
        self.clear()

    def resize(self, new_size: np.int_):
        """Resize the heap to be either larger or smaller"""
        size = self.data.shape[0]
        new_data = np.zeros(new_size, dtype=NodeHeapData)

        if size > 0 and new_size > 0:
            for i in range(min(size, new_size)):
                new_data[i] = self.data[i]

        if new_size < size:
            self.n = new_size

        self.data = new_data

    def push(self, data: NodeHeapData):
        """Push a new item onto the heap"""
        self.n += 1
        if self.n > self.data.shape[0]:
            self.resize(2 * self.n)

        # put the new element at the end,
        # and then perform swaps until the heap is in order
        i = self.n - 1
        self.data[i] = data
        while i > 0:
            i_parent = (i - 1) // 2
            if self.data[i_parent].val <= self.data[i].val:
                break
            else:
                NodeHeap.swap_nodes(self.data, i, i_parent)
                i = i_parent

    def peek(self):
        """Peek at the root of the heap, without removing it"""
        return self.data[0]

    def pop(self) -> NodeHeapData:
        """Remove the root of the heap, and update the remaining nodes"""
        if self.n == 0:
            raise ValueError('cannot pop on empty heap')

        popped_el = self.data[0]

        # pop off the first element, move the last element to the front,
        # and then perform swaps until the heap is back in order
        self.data[0] = self.data[self.n - 1]
        self.n -= 1

        i = 0
        while i < self.n:
            i_child1 = 2 * i + 1
            i_child2 = 2 * i + 2
            i_swap = 0

            if i_child2 < self.n:
                if self.data[i_child1].val <= self.data[i_child2].val:
                    i_swap = i_child1
                else:
                    i_swap = i_child2
            elif i_child1 < self.n:
                i_swap = i_child1
            else:
                break

            if (i_swap > 0) and (self.data[i_swap].val <= self.data[i].val):
                NodeHeap.swap_nodes(self.data, i, i_swap)
                i = i_swap
            else:
                break

        return popped_el

    def clear(self):
        """Clear the heap"""
        self.n = 0

    @staticmethod
    def swap_nodes(arr: np.ndarray, i1: np.int_, i2: np.int_):
        tmp = arr[i1]
        arr[i1] = arr[i2]
        arr[i2] = tmp


class BinaryTree(ABC):
    def __init__(self,
                 galaxies: Sequence[Galaxy],
                 clusterer,
                 leaf_size=40,
                 sample_weight=None,
                 ):
        """
        :param galaxies: Galaxies to be put in tree
        :param clusterer: Clusterer to calculate the distance between galaxies
        :param leaf_size: Number of points at which to switch to brute-force
        :param sample_weight: TODO
        """
        self.data = np.array(tuple(map(lambda el: el.split_coordinates, galaxies)), dtype=np.float_)
        self._clusterer = clusterer
        self.leaf_size = leaf_size
        self._sample_weight = sample_weight
        self.galaxies = galaxies

        self.node_bounds = np.empty((1, 1, 1), dtype=np.float_)  # defined in allocate_data

        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]

        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points between leaf_size and 2 * leaf_size
        self.n_levels = int(np.log2(max(1., (n_samples - 1) / self.leaf_size)) + 1)
        self.n_nodes = (2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(n_samples, dtype=np.int_)
        self.node_data = np.zeros(self.n_nodes, dtype=NodeData)

        self._update_sample_weight(n_samples, sample_weight)

        self.allocate_data(self.n_nodes, n_features)
        self._recursive_build(0, 0, n_samples)
    """Balanced binary tree, used for KD Tree and Ball Tree implementations"""

    @abstractmethod
    def allocate_data(self, n_nodes: np.int_, n_features: np.int_):
        """Allocate tree-specific data"""
        pass

    @abstractmethod
    def init_node(self, i_node: np.int_, idx_start: np.int_, idx_end: np.int_):
        """Initialize the node for the dataset stored in self.data"""
        pass

    def rdist(self, p1: Galaxy, p2: Galaxy) -> np.float_:
        return np.float_(self._clusterer.reduced_distance(p1, p2))

    @abstractmethod
    def min_rdist(self, i_node: np.int_, pt: Galaxy) -> np.float_:
        """Compute the minimum reduced-distance between a point and a node"""
        pass

    @staticmethod
    @abstractmethod
    def min_rdist_dual(tree1: BinaryTree, i_node1: np.int_, tree2: BinaryTree, i_node2: np.int_) -> np.float_:
        """Compute the minimum reduced distance between two nodes"""
        pass

    def rdist_to_dist(self, distances: np.ndarray) -> np.ndarray:
        """Convert reduced distances to real"""
        return self._clusterer.reduced_dist_to_dist(distances)

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
            i_max = find_node_split_dim(self.data, self.idx_array[idx_start:idx_end])
            nts_element(self.data[:, i_max], self.idx_array, idx_start, idx_end)
            self._recursive_build(2 * i_node + 1, idx_start, idx_start + n_mid)
            self._recursive_build(2 * i_node + 2, idx_start + n_mid, idx_end)

    def query(self, points: Sequence[Galaxy], k=1, return_distance=True,
              dualtree=False, breadth_first=False, sort_results=True):
        """
        Query the tree for the k nearest neighbors
        :param points: Points to query neighbours
        :param k: The number of nearest neighbors to return
        :param return_distance:
            if True, return a tuple (d, i) of distances and indices
            if False, return array i
        :param dualtree:
            if True, use the dual tree formalism for the query: a tree is
            built for the query points, and the pair of trees is used to
            efficiently search this space.  This can lead to better
            performance as the number of points grows large.
        :param breadth_first:
            if True, then query the nodes in a breadth-first manner.
            Otherwise, query the nodes in a depth-first manner.
        :param sort_results:
            if True, then distances and indices of each point are sorted
            on return, so that the first column contains the closest points.
            Otherwise, neighbors are returned in an arbitrary order.
        :return:
            i       : if return_distance == False
            (d,i)   : if return_distance == True

            d : ndarray of shape points.shape[:-1] + (k,), dtype=double
                Each entry gives the list of distances to the neighbors of the
                corresponding point.

            i : ndarray of shape points.shape[:-1] + (k,), dtype=int
                Each entry gives the list of indices of neighbors of the
                corresponding point.
        """
        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal to the number of galaxies")

        # initialize heap for neighbors
        heap = NeighborsHeap(len(points), k)

        # node heap for breadth-first queries
        node_heap = None
        if breadth_first:
            node_heap = NodeHeap(self.data.shape[0] // self.leaf_size)

        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0

        if dualtree:
            other = self.__class__(points, clusterer=self._clusterer, leaf_size=self.leaf_size)
            if breadth_first:
                self._query_dual_breadthfirst(other, heap, node_heap)
            else:
                reduced_dist_LB = self.min_rdist_dual(self, 0, other, 0)
                bounds = np.full(other.node_data.shape[0], np.inf)
                self._query_dual_depthfirst(0, other, 0, bounds, heap, reduced_dist_LB)
        else:
            if breadth_first:
                for i in range(len(points)):
                    self._query_single_breadthfirst(points[i], i, heap, node_heap)
            else:
                for i in range(len(points)):
                    reduced_dist_LB = self.min_rdist(0, points[i])
                    self._query_single_depthfirst(0, points[i], i, heap, reduced_dist_LB)

        distances, indices = heap.get_arrays(sort=sort_results)
        distances = self.rdist_to_dist(distances)

        # deflatten results
        if return_distance:
            return distances.reshape((len(points), k)), indices.reshape((len(points), k))
        else:
            return indices.reshape((len(points), k))

    def _query_single_depthfirst(self, i_node: np.int_, pt: Galaxy, i_pt: np.int_,
                                 heap: NeighborsHeap, reduced_dist_LB: np.float_):
        """Recursive Single-tree k-neighbors query, depth-first approach TODO: comments"""
        node_info = self.node_data[i_node]

        # ------------------------------------------------------------
        # Case 1: query point is outside node radius:
        #         trim it from the query
        if reduced_dist_LB > heap.largest(i_pt):
            self.n_trims += 1

        # ------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif node_info.is_leaf:
            self.n_leaves += 1
            for i in range(node_info.idx_start, node_info.idx_end):
                dist_pt = self.rdist(pt, self.galaxies[self.idx_array[i]])
                heap.push(i_pt, dist_pt, self.idx_array[i])

        # ------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the closest
        else:
            self.n_splits += 1
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            reduced_dist_LB_1 = self.min_rdist(i1, pt)
            reduced_dist_LB_2 = self.min_rdist(i2, pt)

            # recursively query subnodes
            if reduced_dist_LB_1 <= reduced_dist_LB_2:
                self._query_single_depthfirst(i1, pt, i_pt, heap, reduced_dist_LB_1)
                self._query_single_depthfirst(i2, pt, i_pt, heap, reduced_dist_LB_2)
            else:
                self._query_single_depthfirst(i2, pt, i_pt, heap, reduced_dist_LB_2)
                self._query_single_depthfirst(i1, pt, i_pt, heap, reduced_dist_LB_1)

    def _query_single_breadthfirst(self, pt: Galaxy, i_pt: np.int_, heap: NeighborsHeap, nodeheap: NodeHeap):
        """Non-recursive single-tree k-neighbors query, breadth-first search TODO: comments"""

        # Set up the node heap and push the head node onto it
        nodeheap_item = NodeHeapData(self.min_rdist(0, pt), 0, 0)
        nodeheap.push(nodeheap_item)

        while nodeheap.n > 0:
            nodeheap_item = nodeheap.pop()
            reduced_dist_LB = nodeheap_item.val
            i_node = nodeheap_item.i1

            # ------------------------------------------------------------
            # Case 1: query point is outside node radius:
            #         trim it from the query
            if reduced_dist_LB > heap.largest(i_pt):
                self.n_trims += 1

            # ------------------------------------------------------------
            # Case 2: this is a leaf node.  Update set of nearby points
            elif self.node_data[i_node].is_leaf:
                self.n_leaves += 1
                for i in range(self.node_data[i_node].idx_start, self.node_data[i_node].idx_end):
                    dist_pt = self.rdist(pt, self.galaxies[self.idx_array[i]])
                    heap.push(i_pt, dist_pt, self.idx_array[i])

            # ------------------------------------------------------------
            # Case 3: Node is not a leaf.  Add subnodes to the node heap
            else:
                self.n_splits += 1
                for i in range(2 * i_node + 1, 2 * i_node + 3):
                    dist = self.min_rdist(i, pt)
                    item = NodeHeapData(dist, i, 0)
                    nodeheap.push(item)

    def _query_dual_depthfirst(self, i_node1: np.int_, other: BinaryTree, i_node2: np.int_, bounds: np.ndarray,
                               heap: NeighborsHeap, reduced_dist_LB: np.float_):
        """
        Recursive dual-tree k-neighbors query, depth-first TODO: comment
        :param i_node1:
        :param other:
        :param i_node2:
        :param bounds:
            Maintained such that bounds[i] is the largest distance among any of the current neighbors
            in node i of the other tree.
        :param heap:
        :param reduced_dist_LB:
        """
        node_info1 = self.node_data[i_node1]
        node_info2 = other.node_data[i_node2]

        data1 = self.galaxies
        data2 = other.galaxies

        # ------------------------------------------------------------
        # Case 1: nodes are further apart than the current bound:
        #         trim both from the query
        if reduced_dist_LB > bounds[i_node2]:
            pass

        # ------------------------------------------------------------
        # Case 2: both nodes are leaves:
        #         do a brute-force search comparing all pairs
        elif node_info1.is_leaf and node_info2.is_leaf:
            bounds[i_node2] = 0

            for i2 in range(node_info2.idx_start, node_info2.idx_end):
                i_pt = other.idx_array[i2]

                if heap.largest(i_pt) <= reduced_dist_LB:
                    continue

                for i1 in range(node_info1.idx_start, node_info1.idx_end):
                    dist_pt = self.rdist(data1[self.idx_array[i1]], data2[i_pt])
                    heap.push(i_pt, dist_pt, self.idx_array[i1])

                # keep track of node bound
                bounds[i_node2] = max(bounds[i_node2], heap.largest(i_pt))

            # update bounds up the tree
            while i_node2 > 0:
                i_parent = (i_node2 - 1) // 2
                bound_max = max(bounds[2 * i_parent + 1], bounds[2 * i_parent + 2])
                if bound_max < bounds[i_parent]:
                    bounds[i_parent] = bound_max
                    i_node2 = i_parent
                else:
                    break

        # ------------------------------------------------------------
        # Case 3a: node 1 is a leaf or is smaller: split node 2 and
        #          recursively query, starting with the nearest subnode
        elif node_info1.is_leaf or (not node_info2.is_leaf and node_info2.radius > node_info1.radius):
            reduced_dist_LB1 = self.__class__.min_rdist_dual(self, i_node1, other, 2 * i_node2 + 1)
            reduced_dist_LB2 = self.__class__.min_rdist_dual(self, i_node1, other, 2 * i_node2 + 2)

            if reduced_dist_LB1 < reduced_dist_LB2:
                self._query_dual_depthfirst(i_node1, other, 2 * i_node2 + 1, bounds, heap, reduced_dist_LB1)
                self._query_dual_depthfirst(i_node1, other, 2 * i_node2 + 2, bounds, heap, reduced_dist_LB2)
            else:
                self._query_dual_depthfirst(i_node1, other, 2 * i_node2 + 2, bounds, heap, reduced_dist_LB2)
                self._query_dual_depthfirst(i_node1, other, 2 * i_node2 + 1, bounds, heap, reduced_dist_LB1)

        # ------------------------------------------------------------
        # Case 3b: node 2 is a leaf or is smaller: split node 1 and
        #          recursively query, starting with the nearest subnode
        else:
            reduced_dist_LB1 = self.__class__.min_rdist_dual(self, 2 * i_node1 + 1, other, i_node2)
            reduced_dist_LB2 = self.__class__.min_rdist_dual(self, 2 * i_node1 + 2, other, i_node2)

            if reduced_dist_LB1 < reduced_dist_LB2:
                self._query_dual_depthfirst(2 * i_node1 + 1, other, i_node2, bounds, heap, reduced_dist_LB1)
                self._query_dual_depthfirst(2 * i_node1 + 2, other, i_node2, bounds, heap, reduced_dist_LB2)
            else:
                self._query_dual_depthfirst(2 * i_node1 + 2, other, i_node2, bounds, heap, reduced_dist_LB2)
                self._query_dual_depthfirst(2 * i_node1 + 1, other, i_node2, bounds, heap, reduced_dist_LB1)

    def _query_dual_breadthfirst(self, other: BinaryTree, heap: NeighborsHeap, nodeheap: NodeHeap):
        """Non-recursive dual-tree k-neighbors query, breadth-first TODO: comments"""
        bounds = np.full(other.node_data.shape[0], np.inf)
        node_data1 = self.node_data
        node_data2 = other.node_data
        data1 = self.galaxies
        data2 = other.galaxies

        # Set up the node heap and push the head nodes onto it
        nodeheap_item = NodeHeapData(self.__class__.min_rdist_dual(self, 0, other, 0), 0, 0)
        nodeheap.push(nodeheap_item)

        while nodeheap.n > 0:
            nodeheap_item = nodeheap.pop()
            reduced_dist_LB = nodeheap_item.val
            i_node1 = nodeheap_item.i1
            i_node2 = nodeheap_item.i2

            node_info1 = node_data1[i_node1]
            node_info2 = node_data2[i_node2]

            # ------------------------------------------------------------
            # Case 1: nodes are further apart than the current bound:
            #         trim both from the query
            if reduced_dist_LB > bounds[i_node2]:
                pass

            # ------------------------------------------------------------
            # Case 2: both nodes are leaves:
            #         do a brute-force search comparing all pairs
            elif node_info1.is_leaf and node_info2.is_leaf:
                bounds[i_node2] = -1

                for i2 in range(node_info2.idx_start, node_info2.idx_end):
                    i_pt = other.idx_array[i2]

                    if heap.largest(i_pt) <= reduced_dist_LB:
                        continue

                    for i1 in range(node_info1.idx_start, node_info1.idx_end):
                        dist_pt = self.rdist(data1[self.idx_array[i1]], data2[i_pt])
                        heap.push(i_pt, dist_pt, self.idx_array[i1])

                    # keep track of node bound
                    bounds[i_node2] = max(bounds[i_node2], heap.largest(i_pt))

            # ------------------------------------------------------------
            # Case 3a: node 1 is a leaf or is smaller: split node 2 and
            #          recursively query, starting with the nearest subnode
            elif node_info1.is_leaf or (not node_info2.is_leaf and (node_info2.radius > node_info1.radius)):
                for i2 in range(2 * i_node2 + 1, 2 * i_node2 + 3):
                    dist = self.__class__.min_rdist_dual(self, i_node1, other, i2)
                    item = NodeHeapData(dist, i1=i_node1, i2=i2)
                    nodeheap.push(item)

            # ------------------------------------------------------------
            # Case 3b: node 2 is a leaf or is smaller: split node 1 and
            #          recursively query, starting with the nearest subnode
            else:
                for i1 in range(2 * i_node1 + 1, 2 * i_node1 + 3):
                    dist = self.__class__.min_rdist_dual(self, i1, other, i_node2)
                    item = NodeHeapData(dist, i1=i1, i2=i_node2)
                    nodeheap.push(item)
