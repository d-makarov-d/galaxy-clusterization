import numpy as np
import unittest

from data_structures.binary_tree import NeighborsHeap, NodeHeap, NodeHeapData


def nodeheap_sort(vals: np.ndarray):
    """In-place reverse sort of vals using NodeHeap"""
    indices = np.zeros(vals.shape[0], dtype=np.int_)
    vals_sorted = np.zeros_like(vals)

    # use initial size 0 to check corner case
    heap = NodeHeap(0)
    for i in range(vals.shape[0]):
        data = NodeHeapData(vals[i], i, i+1)
        data.val = vals[i]
        data.i1 = i
        data.i2 = i + 1
        heap.push(data)

    for i in range(vals.shape[0]):
        data = heap.pop()
        vals_sorted[i] = data.val
        indices[i] = data.i1

    return np.asarray(vals_sorted), np.asarray(indices)


class Heaps(unittest.TestCase):
    def test_neighbors_heap(self):
        n_pts = 5
        n_nbrs = 10

        heap = NeighborsHeap(n_pts, n_nbrs)
        rng = np.random.RandomState(0)

        for row in range(n_pts):
            d_in = rng.random_sample(2 * n_nbrs)
            i_in = np.arange(2 * n_nbrs, dtype=np.int_)
            for d, i in zip(d_in, i_in):
                heap.push(row, d, i)

            ind = np.argsort(d_in)
            d_in = d_in[ind]
            i_in = i_in[ind]

            d_heap, i_heap = heap.get_arrays(sort=True)

            self.assertListEqual(list(d_in[:n_nbrs]), list(d_heap[row]))
            self.assertListEqual(list(i_in[:n_nbrs]), list(i_heap[row]))

    def test_node_heap(self):
        n_nodes = 50
        rng = np.random.RandomState(0)
        vals = rng.random_sample(n_nodes)

        i1 = np.argsort(vals)
        vals2, i2 = nodeheap_sort(vals)

        self.assertListEqual(list(i1), list(i2))
        self.assertListEqual(list(vals[i1]), list(vals2))
