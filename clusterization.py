from __future__ import annotations

from abc import ABC, abstractmethod
from math import inf
import numpy as np
from functools import reduce
from collections.abc import Iterable
from typing import Callable, Tuple, Union

from db.galaxy import Galaxy
from hdbscan.hdbscan import hdbscan_prims_kdtree, _tree_to_labels


class Cluster(Galaxy, ABC):
    def __init__(self, members: tuple[Galaxy]):
        """
        Describes some galaxies, counted as one object
        :param members: Galaxies or clusters, contained by this object
        """
        self._members = members

        params = self.merge(members)
        super().__init__(*params)

    @property
    def members(self) -> tuple[Galaxy]:
        return self._members

    @property
    def galaxies(self) -> tuple[Galaxy]:
        return tuple(
            reduce(lambda acc, el: acc + (el.galaxies if isinstance(el, Cluster) else (el,)), self.members, tuple()))

    @abstractmethod
    def merge(self, members: tuple[Galaxy]) -> Tuple[float, float, float, float, float]:
        """Merges individual member parameters to cluster parameters"""
        pass


class Clusterer(ABC):
    @abstractmethod
    def calc_distance(self, a: Galaxy, b: Galaxy) -> float:
        """Calculate distance for clustering"""
        pass

    def reduced_distance(self, a: Galaxy, b: Galaxy) -> float:
        """Calculate reduced distance, e/.g squared distance for euclidean metric. Standard distance for default"""
        return self.calc_distance(a, b)

    def reduced_dist_to_dist(self, distances: np.ndarray) -> np.ndarray:
        return distances

    @property
    @abstractmethod
    def cluster_class(self) -> Cluster.__class__:
        """Returns class for cluster instantiation"""
        pass

    def hierarchical(self, galaxies: Iterable[Galaxy], criterion: Callable[[Cluster, Cluster], bool]) -> list[Cluster]:
        """
        Produces ierarhical tree of clusters. O(n^2) complexity
        :param galaxies: Objects for clustering
        :param criterion: Clustering border criteria, cluster is made only if it returns true
        :return: Root cluster
        """
        clusters: list[Cluster] = list(galaxies)
        # maps each cluster to its distances
        distances = dict(map(lambda c: (c, {}), clusters))
        min_dist = (inf, (None, None))

        def merge(a: Cluster, b: Cluster) -> tuple:
            """
            Merge 2 clusters
            :param a: 1-st cluster
            :param b: 2-nd cluster
            :return: new min_distance, containing minimal distance value and pair of clusters for this distance
            """
            merged = self.cluster_class((a, b))
            clusters.remove(a)
            clusters.remove(b)
            if a in distances:
                del distances[a]
            if b in distances:
                del distances[b]
            distances[merged] = {}
            global_min = (inf, (None, None))
            for cluster in clusters:
                # Is row minimum needs to be recalculated
                recalc_min = False
                if a in distances[cluster]:
                    del distances[cluster][a]
                    if a in distances[cluster]['min'][1]:
                        recalc_min = True
                if b in distances[cluster]:
                    del distances[cluster][b]
                    if b in distances[cluster]['min'][1]:
                        recalc_min = True
                d = self.calc_distance(cluster, merged)
                if recalc_min:
                    local_min = (inf, (None, None))
                    for key in distances[cluster]:
                        if key == 'min':
                            continue
                        if distances[cluster][key] < d:
                            local_min = (dist, (cluster, key))
                    distances[cluster]['min'] = local_min
                distances[cluster][merged] = d
                distances[merged] = {'min': (inf, (None, None))}
                if d < distances[cluster]['min'][0]:
                    distances[cluster]['min'] = (d, (cluster, merged))
                if distances[cluster]['min'][0] < global_min[0]:
                    global_min = distances[cluster]['min']
            clusters.append(merged)
            return global_min

        # calculate each - to each distance
        for i in range(0, len(clusters)):
            local_min = (inf, (None, None))
            for j in range(i + 1, len(clusters)):
                dist = self.calc_distance(clusters[i], clusters[j])
                distances[clusters[i]][clusters[j]] = dist
                if dist < min_dist[0]:
                    min_dist = (dist, (clusters[i], clusters[j]))
                if dist < local_min[0]:
                    local_min = (dist, (clusters[i], clusters[j]))
            distances[clusters[i]]['min'] = local_min
        # repeat merging closest clusters
        while len(clusters) > 1 and criterion(*min_dist[1]):
            min_dist = merge(*min_dist[1])

        return clusters

    def hdbscan(self, galaxies: Union[Iterable[Galaxy], np.ndarray], min_cluster_size=5,
                min_samples=None, alpha=1.0, cluster_selection_epsilon=0.0,
                max_cluster_size=0, leaf_size=40, algorithm='best',
                approx_min_span_tree=True, gen_min_span_tree=False, core_dist_n_jobs=4,
                cluster_selection_method='eom', allow_single_cluster=False):
        """ Fork of HDBSCAN implementation in repository (https://github.com/scikit-learn-contrib/hdbscan/)
            Not much changes, made only to support project's data structures and specific astrological distance
            metric.
            Perform HDBSCAN clustering from a vector array or distance matrix.

            Parameters
            ----------
            galaxies : Collection of galaxies to be clustered or array of distances between samples if
                ``metric='precomputed'``.

            min_cluster_size : int, optional (default=5)
                The minimum number of samples in a group for that group to be
                considered a cluster; groupings smaller than this size will be left
                as noise.

            min_samples : int, optional (default=None)
                The number of samples in a neighborhood for a point
                to be considered as a core point. This includes the point itself.
                defaults to the min_cluster_size.

            cluster_selection_epsilon: float, optional (default=0.0)
                A distance threshold. Clusters below this value will be merged.

            alpha : float, optional (default=1.0)
                A distance scaling parameter as used in robust single linkage.
                See [2]_ for more information.

            max_cluster_size : int, optional (default=0)
                A limit to the size of clusters returned by the eom algorithm.
                Has no effect when using leaf clustering (where clusters are
                usually small regardless) and can also be overridden in rare
                cases by a high value for cluster_selection_epsilon. Note that
                this should not be used if we want to predict the cluster labels
                for new points in future (e.g. using approximate_predict), as
                the approximate_predict function is not aware of this argument.

            leaf_size : int, optional (default=40)
                Leaf size for trees responsible for fast nearest
                neighbour queries.

            algorithm : string, optional (default='best')
                Exactly which algorithm to use; hdbscan has variants specialised
                for different characteristics of the data. By default this is set
                to ``best`` which chooses the "best" algorithm given the nature of
                the data. You can force other options if you believe you know
                better. Options are:
                    * ``best``
                    * ``generic``
                    * ``prims_kdtree``
                    * ``prims_balltree``        TODO: implement
                    * ``boruvka_kdtree``        TODO: implement
                    * ``boruvka_balltree``      TODO: implement

            approx_min_span_tree : bool, optional (default=True)
                Whether to accept an only approximate minimum spanning tree.
                For some algorithms this can provide a significant speedup, but
                the resulting clustering may be of marginally lower quality.
                If you are willing to sacrifice speed for correctness you may want
                to explore this; in general this should be left at the default True.

            gen_min_span_tree : bool, optional (default=False)
                Whether to generate the minimum spanning tree for later analysis.

            core_dist_n_jobs : int, optional (default=4)
                Number of parallel jobs to run in core distance computations (if
                supported by the specific algorithm). For ``core_dist_n_jobs``
                below -1, (n_cpus + 1 + core_dist_n_jobs) are used.

            cluster_selection_method : string, optional (default='eom')
                The method used to select clusters from the condensed tree. The
                standard approach for HDBSCAN* is to use an Excess of Mass algorithm
                to find the most persistent clusters. Alternatively you can instead
                select the clusters at the leaves of the tree -- this provides the
                most fine grained and homogeneous clusters. Options are:
                    * ``eom``
                    * ``leaf``

            allow_single_cluster : bool, optional (default=False)
                By default HDBSCAN* will not produce a single cluster, setting this
                to t=True will override this and allow single cluster results in
                the case that you feel this is a valid result for your dataset.
                (default False)

            Returns
            -------
            labels : ndarray, shape (n_samples, )
                Cluster labels for each point.  Noisy samples are given the label -1.

            probabilities : ndarray, shape (n_samples, )
                Cluster membership strengths for each point. Noisy samples are assigned
                0.

            cluster_persistence : array, shape  (n_clusters, )
                A score of how persistent each cluster is. A score of 1.0 represents
                a perfectly stable cluster that persists over all distance scales,
                while a score of 0.0 represents a perfectly ephemeral cluster. These
                scores can be guage the relative coherence of the clusters output
                by the algorithm.

            condensed_tree : record array
                The condensed cluster hierarchy used to generate clusters.

            single_linkage_tree : ndarray, shape (n_samples - 1, 4)
                The single linkage tree produced during clustering in scipy
                hierarchical clustering format
                (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).

            min_spanning_tree : ndarray, shape (n_samples - 1, 3)
                The minimum spanning as an edgelist. If gen_min_span_tree was False
                this will be None.

            References
            ----------

            .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
               Density-based clustering based on hierarchical density estimates.
               In Pacific-Asia Conference on Knowledge Discovery and Data Mining
               (pp. 160-172). Springer Berlin Heidelberg.

            .. [2] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
               cluster tree. In Advances in Neural Information Processing Systems
               (pp. 343-351).

            .. [3] Malzer, C., & Baum, M. (2019). A Hybrid Approach To Hierarchical
               Density-based Cluster Selection. arxiv preprint 1911.02282.
            """

        if min_samples is None:
            min_samples = min_cluster_size

        if type(min_samples) is not int or type(min_cluster_size) is not int:
            raise ValueError('Min samples and min cluster size must be integers!')

        if min_samples <= 0 or min_cluster_size <= 0:
            raise ValueError('Min samples and Min cluster size must be positive integers')

        if min_cluster_size == 1:
            raise ValueError('Min cluster size must be greater than one')

        if type(cluster_selection_epsilon) is int:
            cluster_selection_epsilon = float(cluster_selection_epsilon)

        if type(cluster_selection_epsilon) is not float or cluster_selection_epsilon < 0.0:
            raise ValueError('Epsilon must be a float value greater than or equal to 0!')

        if not isinstance(alpha, float) or alpha <= 0.0:
            raise ValueError('Alpha must be a positive float value greater than 0!')

        if leaf_size < 1:
            raise ValueError('Leaf size must be greater than 0!')

        if cluster_selection_method not in ('eom', 'leaf'):
            raise ValueError('Invalid Cluster Selection Method: %s\n'
                             'Should be one of: "eom", "leaf"\n')

        is_precomputed = not isinstance(galaxies[0], Galaxy)

        min_samples = min(len(galaxies) - 1, min_samples)
        if min_samples == 0:
            min_samples = 1

        if algorithm != 'best':
            if algorithm == 'generic':
                single_linkage_tree, result_min_span_tree = \
                    _hdbscan_generic(min_samples, alpha, leaf_size, gen_min_span_tree)
            elif algorithm == 'prims_kdtree':
                single_linkage_tree, result_min_span_tree = \
                    hdbscan_prims_kdtree(galaxies, self, min_samples, alpha, leaf_size, gen_min_span_tree)
            else:
                raise TypeError('Unknown algorithm type %s specified' % algorithm)
        else:
            # because galaxies live in 3-dimensional euclidean space, KD-tree with boruvka is considered the best method
            # TODO: implement
            single_linkage_tree, result_min_span_tree = \
                hdbscan_prims_kdtree(galaxies, self, min_samples, alpha, leaf_size, gen_min_span_tree)

        return _tree_to_labels(
            single_linkage_tree,
            min_cluster_size,
            cluster_selection_method,
            allow_single_cluster,
            cluster_selection_epsilon,
            max_cluster_size) + (result_min_span_tree,)
