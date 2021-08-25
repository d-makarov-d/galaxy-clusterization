from __future__ import annotations

import unittest
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
import statistics

from clusterization import Clusterer, Cluster
from db.galaxy import Galaxy


class EuclideanCluster(Cluster):
    def merge_z(self, data: Iterable[float]) -> float:
        return statistics.mean(data)

    def merge_alpha(self, data: Iterable[float]) -> float:
        return statistics.mean(data)

    def merge_delta(self, data: Iterable[float]) -> float:
        return statistics.mean(data)

    def merge_lum(self, data: Iterable[float]) -> float:
        return statistics.mean(data)


class EuclideanClusterer(Clusterer):
    """Counts distances between EuclideanCluster's using Euclidean distance on alpha and delta"""
    @property
    def cluster_class(self) -> Cluster.__class__:
        return EuclideanCluster

    def calc_distance(self, a: EuclideanCluster, b: EuclideanCluster) -> float:
        return math.sqrt((a.alpha + b.alpha)**2 + (a.delta + b.delta)**2)


class Hierarchical(unittest.TestCase):
    def test_show_clusters(self):
        data = np.load('tests/clusterable_data.npy')
        samples = list(map(lambda el: Galaxy(0, el[0], el[1], 0), data))
        galaxies = []
        for i in range(0, 300):
            galaxies.append(random.choice(samples))
            samples.remove(galaxies[-1])
        elapsed = time.time()
        root = EuclideanClusterer().hierarchical(galaxies)
        elapsed = time.time() - elapsed
        depth = 3
        clusters = []
        tmp = [root]
        for i in range(1, depth+1):
            for c in tmp:
                if i == depth or not isinstance(c, Cluster):
                    clusters.append(c)
                else:
                    for m in c.members:
                        tmp.append(m)
                    tmp.remove(c)

        for cluster in clusters:
            if isinstance(cluster, Cluster):
                x = tuple(map(lambda g: g.alpha, cluster.galaxies))
                y = tuple(map(lambda g: g.delta, cluster.galaxies))
                plt.scatter(x, y, alpha=0.25, linewidth=0)
            else:
                x = cluster.alpha
                y = cluster.delta
                plt.scatter(x, y, alpha=0.25, linewidth=0)
        plt.title('Hierarchical clustering, %i points, %.3f elapsed time' % (len(data), elapsed))
        plt.show()
