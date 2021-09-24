import numpy as np
from numpy.testing import assert_allclose
import unittest
from astropy.coordinates import cartesian_to_spherical

from data_structures.binary_tree import NeighborsHeap, NodeHeap, NodeHeapData
from data_structures.kd_tree import KDTree
from db.galaxy import Galaxy
from tests.test_visual import Euclidean3Clusterer


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


def pairwise(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xarr = np.asarray(X)
    Yarr = np.asarray(Y)
    Darr = np.zeros((Xarr.shape[0], Yarr.shape[0]))
    for i1 in range(X.shape[0]):
        for i2 in range(Y.shape[0]):
            Darr[i1, i2] = np.sqrt(np.sum((Xarr[i1, :] - Y[i2, :])**2))
    return Darr


def brute_force_neighbors(X, Y, k):
    D = pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


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


class KDTreeTests(unittest.TestCase):
    def test_nn_tree_query(self):
        k = [1, 3, 5]
        dualtree = [True, False]
        breath_first = [True, False]

        def run_test(k: int, dualtree: bool, breadth_first: bool):
            rng = np.random.RandomState(0)
            X = rng.random_sample((40, 3))
            Y = rng.random_sample((10, 3))

            def gal_from_cart(cart):
                r, lat, lon = cartesian_to_spherical(*cart)
                return Galaxy(r.value, lon.value, lat.value, 1, 1)

            galaxies_x = tuple(map(lambda el: gal_from_cart(el), X))
            galaxies_y = tuple(map(lambda el: gal_from_cart(el), Y))
            clusterer = Euclidean3Clusterer()
            tree = KDTree(galaxies_x, clusterer, leaf_size=1)

            dist1, ind1 = tree.query(galaxies_y, k, dualtree=dualtree, breadth_first=breadth_first)
            dist2, ind2 = brute_force_neighbors(X, Y, k)

            assert_allclose(dist2, dist1)

        for K in k:
            for dt in dualtree:
                for bf in breath_first:
                    run_test(K, dt, bf)
