from __future__ import annotations

import unittest
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
import numpy

from numpy.typing import NDArray

from clusterization import Clusterer, Cluster
from db.galaxy import Galaxy


class EuclideanCluster(Cluster):
    def merge(self,
              dist: NDArray, ra: NDArray, dec: NDArray, mass: NDArray, ev: NDArray
              ) -> Tuple[float, float, float, float, float]:
        return(
            numpy.average(dist),
            numpy.average(ra),
            numpy.average(dec),
            numpy.average(mass),
            numpy.average(ev),
        )


class EuclideanClusterer(Clusterer):
    """Counts distances between EuclideanCluster's using Euclidean distance on alpha and delta"""

    @property
    def cluster_class(self) -> Cluster.__class__:
        return EuclideanCluster

    def calc_distance(self, a: EuclideanCluster, b: EuclideanCluster) -> float:
        return math.sqrt((a.ra - b.ra) ** 2 + (a.dec - b.dec) ** 2)


class Hierarchical(unittest.TestCase):
    def test_show_clusters(self):
        data = np.load('tests/clusterable_data.npy')
        """data = [
            [0, 0],
            [1, 0],
            [0, 1],
            [10, 10],
            [11, 10],
            [10, 11],
            [2, 8]
        ]"""
        galaxies = list(map(lambda el: Galaxy(0, el[0], el[1], 0, 0), data))
        random.shuffle(galaxies)
        elapsed = time.time()
        clusters = EuclideanClusterer().hierarchical(galaxies,
                                                     lambda c1, c2: EuclideanClusterer().calc_distance(c1, c2) < 0.5)
        elapsed = time.time() - elapsed
        """root = clusters[0]
        depth = 2
        clusters = []
        tmp = [root]
        for i in range(1, depth+1):
            for c in tmp:
                if i == depth or not isinstance(c, Cluster):
                    clusters.append(c)
                else:
                    for m in c.members:
                        tmp.append(m)
                    tmp.remove(c)"""

        for cluster in clusters:
            if isinstance(cluster, Cluster):
                x = tuple(map(lambda g: g.ra, cluster.galaxies))
                y = tuple(map(lambda g: g.dec, cluster.galaxies))
            else:
                x = cluster.ra
                y = cluster.dec
            plt.scatter(x, y, alpha=0.25, linewidth=0)
        plt.title('Hierarchical clustering, %i points, %.3f elapsed time' % (len(data), elapsed))
        plt.show()
