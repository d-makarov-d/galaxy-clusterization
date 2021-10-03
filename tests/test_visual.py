from __future__ import annotations

import unittest
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random

from clusterization import Clusterer, Cluster
from db.galaxy import Galaxy, GalaxiesDB
from visualization import draw_clusters


class EuclideanCluster(Cluster):
    def merge(self, members: tuple[Galaxy]) -> Tuple[float, float, float, float, float]:
        dist = np.zeros(len(members))
        ra = np.zeros(len(members))
        dec = np.zeros(len(members))
        mass = np.zeros(len(members))
        ev = np.zeros(len(members))
        for i in range(0, len(members)):
            dist[i] = members[i].dist
            ra[i] = members[i].ra
            dec[i] = members[i].dec
            mass[i] = members[i].mass
            ev[i] = members[i].ed
        return(
            np.average(dist),
            np.average(ra),
            np.average(dec),
            np.average(mass),
            np.average(ev),
        )


class EuclideanClusterer(Clusterer):
    """Counts distances between EuclideanCluster's using Euclidean distance on alpha and delta"""

    @property
    def cluster_class(self) -> Cluster.__class__:
        return EuclideanCluster

    def calc_distance(self, a: EuclideanCluster, b: EuclideanCluster) -> float:
        return math.sqrt((a.ra - b.ra) ** 2 + (a.dec - b.dec) ** 2)


class Euclidean3Clusterer(Clusterer):
    """Counts distances between EuclideanCluster's transforming spherical coordinates to cartesian"""

    @property
    def cluster_class(self) -> Cluster.__class__:
        return EuclideanCluster

    def calc_distance(self, a: EuclideanCluster, b: EuclideanCluster) -> float:
        p1 = np.array(a.cart, dtype=float)
        p2 = np.array(b.cart, dtype=float)
        return np.sqrt(np.sum((p2 - p1)**2))


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

    def test_2d_hdbscan(self):
        data = np.load('tests/clusterable_data.npy')
        galaxies = list(map(lambda el: Galaxy(0, el[0], el[1], 0, 0), data))
        random.shuffle(galaxies)

        elapsed = time.time()
        labels, *_ = EuclideanClusterer().hdbscan(
            galaxies,
            algorithm='prims_kdtree',
            min_cluster_size=5,
            cluster_selection_epsilon=0.015
        )
        elapsed = time.time() - elapsed

        coords = np.zeros((len(galaxies), 2))
        for i, galaxy in enumerate(galaxies):
            coords[i, :] = np.array([galaxy.ra, galaxy.dec])
        print(max(labels))
        for i in range(max(labels)):
            mask = labels == i
            plt.scatter(coords[mask, 0], coords[mask, 1], alpha=0.25, linewidth=0)
        noise = labels == -1
        plt.scatter(coords[noise, 0], coords[noise, 1], c='grey', alpha=0.25, linewidth=0)

        plt.title('HDBSCAN clustering, %i points, %.3f elapsed time' % (len(data), elapsed))
        plt.show()

    def test_3d_euclid(self):
        db = GalaxiesDB('tests/test_groups.db')
        gals = db.find(Galaxy)
        clusterer = Euclidean3Clusterer()
        clusters = clusterer.hierarchical(gals, lambda c1, c2: clusterer.calc_distance(c1, c2) < 1)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        draw_clusters(ax, clusters)
        plt.show()
        db.close()
