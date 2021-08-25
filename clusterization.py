from __future__ import annotations

from abc import ABC, abstractmethod
from math import inf
import statistics
from functools import reduce
from collections.abc import Iterable

from db.galaxy import Galaxy


class Cluster(Galaxy, ABC):
    def __init__(self, members: tuple[Galaxy]):
        """
        Describes some galaxies, counted as one object
        :param members: Galaxies or clusters, contained by this object
        """
        self._members = members
        self._z = self.merge_z(map(lambda el: el.z, self.members))
        self._alpha = self.merge_alpha(map(lambda el: el.alpha, self.members))
        self._delta = self.merge_delta(map(lambda el: el.delta, self.members))
        self._lum = self.merge_lum(map(lambda el: el.lum, self.members))
        super().__init__(self.z, self.alpha, self.delta, self.lum)

    @property
    def members(self) -> tuple[Galaxy]:
        return self._members

    @property
    def galaxies(self) -> tuple[Galaxy]:
        return tuple(reduce(lambda acc, el: acc + (el.galaxies if isinstance(el, Cluster) else (el, )), self.members, tuple()))

    @abstractmethod
    def merge_z(self, data: Iterable[float]) -> float:
        pass

    @abstractmethod
    def merge_alpha(self, data: Iterable[float]) -> float:
        pass

    @abstractmethod
    def merge_delta(self, data: Iterable[float]) -> float:
        pass

    @abstractmethod
    def merge_lum(self, data: Iterable[float]) -> float:
        pass


class Clusterer(ABC):
    @abstractmethod
    def calc_distance(self, a: Cluster, b: Cluster) -> float:
        """Calculate distance for clustering"""
        pass

    @property
    @abstractmethod
    def cluster_class(self) -> Cluster.__class__:
        """Returns class for cluster instantiation"""
        pass

    def hierarchical(self, galaxies: list[Galaxy]) -> Cluster:
        """
        Produces ierarhical tree of clusters. O(n^2) complexity
        :param galaxies: Objects for clusterization
        :return: Root cluster
        """
        clusters: list[Cluster] = galaxies # list(map(lambda g: Cluster((g, )), galaxies))
        # maps each cluster to its distances
        distances = dict(map(lambda c: (c, {}), clusters))
        min_dist = (inf, (None, None))

        def merge(a: Cluster, b: Cluster):
            merged = self.cluster_class((a, b))
            clusters.remove(a)
            clusters.remove(b)
            if a in distances:
                del distances[a]
            if b in distances:
                del distances[b]
            distances[merged] = {}
            for cluster in clusters:
                if a in distances[cluster]:
                    del distances[cluster][a]
                if b in distances[cluster]:
                    del distances[cluster][b]
                distances[cluster][merged] = self.calc_distance(cluster, merged)
            clusters.append(merged)

        # calculate each - to each distance
        for i in range(0, len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = self.calc_distance(clusters[i], clusters[j])
                distances[clusters[i]][clusters[j]] = dist
                if dist < min_dist[0]:
                    min_dist = (dist, (clusters[i], clusters[j]))
        # repeat merging closest clusters
        while len(clusters) > 1:
            merge(*min_dist[1])
            min_dist = (inf, (None, None))
            for i in range(0, len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = distances[clusters[i]][clusters[j]]
                    if dist < min_dist[0]:
                        min_dist = (dist, (clusters[i], clusters[j]))

        return clusters[0]
