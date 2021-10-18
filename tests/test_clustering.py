import unittest

import numpy as np

from clusterization import Clusterer, Cluster
from tests.test_visual import EuclideanCluster
from db.galaxy import Galaxy

X = np.load('tests/data/hdbscan_test_set.npy')


class EuclideanClusterer(Clusterer):
    """Counts distances between EuclideanCluster's using Euclidean distance on mass and ed"""

    @property
    def cluster_class(self) -> Cluster.__class__:
        return EuclideanCluster

    def calc_distance(self, a: EuclideanCluster, b: EuclideanCluster) -> float:
        return np.sqrt((a.mass - b.mass) ** 2 + (a.ed - b.ed) ** 2)


class Hdbscan(unittest.TestCase):
    def test_hdbscan_prims_kdtree(self):
        galaxies = list(map(lambda row: Galaxy(1, 0, 0, row[0], row[1]), X))
        clusterer = EuclideanClusterer()
        labels, p, persist, ctree, ltree, mtree = clusterer.hdbscan(galaxies, algorithm='prims_kdtree')

        n_clusters = len(set(labels)) - int(-1 in labels)
        self.assertEqual(n_clusters, 3)

