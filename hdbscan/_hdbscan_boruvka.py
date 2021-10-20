import numpy as np
from joblib import Parallel, delayed
from typing import Sequence

from data_structures.kd_tree import KDTree
from db.galaxy import Galaxy


class BoruvkaUnionFind:
    """Efficient union find implementation. TODO in C"""
    def __init__(self, size: int):
        """
        :param size: The total size of the set of objects to track via the union find structure.
        """
        self._parent = np.arange(size, dtype=np.int_)
        self._rank = np.zeros(size, dtype=np.uint8)
        self.is_component = np.ones(size, dtype=bool)

    def union_(self, x: np.int_, y: np.int_):
        """Union together elements x and y"""
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return 0

        if self._rank[x_root] < self._rank[y_root]:
            self._parent[x_root] = y_root
            self.is_component[x_root] = False
        elif self._rank[x_root] > self._rank[y_root]:
            self._parent[y_root] = x_root
            self.is_component[y_root] = False
        else:
            self._rank[x_root] += 1
            self._parent[y_root] = x_root
            self.is_component[y_root] = False

    def find(self, x: np.int_) -> np.int_:
        """Find the root or identifier for the component that x is in"""

        x_parent = self._parent[x]
        while True:
            if x_parent == x:
                return x
            x_grandparent = self._parent[x_parent]
            self._parent[x] = x_grandparent
            x = x_parent
            x_parent = x_grandparent

    def components(self) -> np.ndarray:
        """Return an array of all component roots/identifiers"""
        return self.is_component.nonzero()[0]


def _core_dist_query(tree, galaxies: Sequence[Galaxy], min_samples: int):
    return tree.query(galaxies, k=min_samples, dualtree=True, breadth_first=True)


class KDTreeBoruvkaAlgorithm:
    """Dual tree Boruvka Algorithm"""
    def __init__(self, tree: KDTree, min_samples=5, leaf_size=20, alpha=1.,
                 approx_min_span_tree=False, n_jobs=4):
        """
        :param tree: The kd-tree to run Dual Tree Boruvka over.
        :param min_samples: The min_samples parameter of HDBSCAN used to determine core distances.
        :param leaf_size:
            The Boruvka algorithm benefits from a smaller leaf size than
            standard kd-tree nearest neighbor searches. The tree passed in
            is used for a kNN search for core distance. A second tree is
            constructed with a smaller leaf size for Boruvka; this is that
            leaf size.
        :param alpha: The alpha distance scaling parameter as per Robust Single Linkage.
        :param approx_min_span_tree:
            Take shortcuts and only approximate the min spanning tree.
            This is considerably faster but does not return a true
            minimal spanning tree.
        :param n_jobs: The number of parallel jobs used to compute core distances.
        """
        self.core_dist_tree = tree
        self._raw_galaxies = tree.galaxies
        self.tree = KDTree(tree.galaxies, tree.clusterer, leaf_size)
        self.n_jobs = n_jobs
        self.min_samples = min_samples
        self.clusterer = tree.clusterer
        self.approx_min_span_tree = approx_min_span_tree
        self.alpha = alpha

        self.num_points = len(self.tree.galaxies)
        self.num_nodes = self.tree.node_data.shape[0]

        self.components = np.arange(self.num_points)
        self.bounds = np.empty(self.num_nodes, np.double)
        self.component_of_point = np.empty(self.num_points, dtype=np.intp)
        self.component_of_node = np.empty(self.num_nodes, dtype=np.intp)
        self.candidate_neighbor = np.empty(self.num_points, dtype=np.intp)
        self.candidate_point = np.empty(self.num_points, dtype=np.intp)
        self.candidate_distance = np.empty(self.num_points, dtype=np.double)
        self.component_union_find = BoruvkaUnionFind(self.num_points)

        self.edges = np.empty((self.num_points - 1, 3))
        self.num_edges = 0

        self.idx_array = self.tree.idx_array
        self.node_data = self.tree.node_data

        self._initialize_components()
        self._compute_bounds()

    def _compute_bounds(self):
        """Initialize core distances"""
        # A shortcut: if we have a lot of points then we can split the points
        # into four piles and query them in parallel. On multicore systems
        # (most systems) this amounts to a 2x-3x wall clock improvement.
        if self.num_points > 16384 and self.n_jobs > 1:
            split_cnt = self.num_points // self.n_jobs
            datasets = []
            for i in range(self.n_jobs):
                if i == self.n_jobs - 1:
                    datasets.append(np.asarray(self.tree.galaxies[i * split_cnt:]))
                else:
                    datasets.append(np.asarray(self.tree.galaxies[i * split_cnt:(i + 1) * split_cnt]))

            knn_data = Parallel(n_jobs=self.n_jobs)(
                delayed(_core_dist_query)
                (self.core_dist_tree, points, self.min_samples + 1)
                for points in datasets
            )
            knn_dist = np.vstack([x[0] for x in knn_data])
            knn_indices = np.vstack([x[1] for x in knn_data])
        else:
            knn_dist, knn_indices = self.core_dist_tree.query(
                self.tree.galaxies,
                k=self.min_samples + 1,
                dualtree=True,
                breadth_first=True
            )

        self.core_distance = knn_dist[:, self.min_samples].copy()

        # Since we do everything in terms of rdist to free up the GIL
        # we need to convert all the core distances beforehand
        # to make comparison feasible.
        self.core_distance = self.clusterer.dist_to_reduced(self.core_distance)

        # Since we already computed NN distances for the min_samples closest
        # points we can use this to do the first round of boruvka -- we won't
        # get every point due to core_distance/mutual reachability distance
        # issues, but we'll get quite a few, and they are the hard ones to
        # get, so fill in any we can and then run update components.
        for n in range(self.num_points):
            for i in range(1, self.min_samples + 1):
                m = np.int_(knn_indices[n, i])
                if self.core_distance[m] <= self.core_distance[n]:
                    self.candidate_point[n] = n
                    self.candidate_neighbor[n] = m
                    self.candidate_distance[n] = self.core_distance[n]
                    break

        self.update_components()

        for n in range(self.num_nodes):
            self.bounds[n] = np.inf

    def _initialize_components(self):
        """Initialize components of the min spanning tree (eventually there
        is only one component; initially each point is its own component)"""

        for n in range(self.num_points):
            self.component_of_point[n] = n
            self.candidate_neighbor[n] = -1
            self.candidate_point[n] = -1
            self.candidate_distance[n] = np.inf

        for n in range(self.num_nodes):
            self.component_of_node[n] = -(n+1)

    def update_components(self) -> int:
        """Having found the nearest neighbor not in the same component for
        each current component (via tree traversal), run through adding
        edges to the min spanning tree and recomputing components via
        union find."""
        # For each component there should be a:
        #   - candidate point (a point in the component)
        #   - candiate neighbor (the point to join with)
        #   - candidate_distance (the distance from point to neighbor)
        #
        # We will go through and and an edge to the edge list
        # for each of these, and the union the two points
        # together in the union find structure
        for c in range(self.components.shape[0]):
            component = self.components[c]
            source = self.candidate_point[component]
            sink = self.candidate_neighbor[component]
            if source == -1 or sink == -1:
                continue
                # raise ValueError('Source or sink of edge is not defined!')
            current_source_component = self.component_union_find.find(source)
            current_sink_component = self.component_union_find.find(sink)
            if current_source_component == current_sink_component:
                # We've already joined these, so ignore this edge
                self.candidate_point[component] = -1
                self.candidate_neighbor[component] = -1
                self.candidate_distance[component] = np.inf
                continue
            self.edges[self.num_edges, 0] = source
            self.edges[self.num_edges, 1] = sink
            self.edges[self.num_edges, 2] = self.clusterer.reduced_dist_to_dist(self.candidate_distance[component])
            self.num_edges += 1

            self.component_union_find.union_(source, sink)

            # Reset everything,and check if we're done
            self.candidate_distance[component] = np.inf
            if self.num_edges == self.num_points - 1:
                self.components = self.component_union_find.components()
                return self.components.shape[0]

        # After having joined everything in the union find data
        # structure we need to go through and determine the components
        # of each point for easy lookup.
        #
        # Have done that we then go through and set the component
        # of each node, as this provides fast pruning in later
        # tree traversals.
        for n in range(self.tree.data.shape[0]):
            self.component_of_point[n] = self.component_union_find.find(n)

        for n in range(self.tree.node_data.shape[0] - 1, -1, -1):
            node_info = self.node_data[n]
            # Case 1:
            #    If the node is a leaf we need to check that every point
            #    in the node is of the same component
            if node_info.is_leaf:
                current_component = self.component_of_point[self.idx_array[node_info.idx_start]]
                for i in range(node_info.idx_start + 1, node_info.idx_end):
                    p = self.idx_array[i]
                    if self.component_of_point[p] != current_component:
                        break
                else:
                    self.component_of_node[n] = current_component
            # Case 2:
            #    If the node is not a leaf we only need to check
            #    that both child nodes are in the same component
            else:
                child1 = 2 * n + 1
                child2 = 2 * n + 2
                if self.component_of_node[child1] == self.component_of_node[child2]:
                    self.component_of_node[n] = self.component_of_node[child1]

        # Since we're working with mutual reachability distance we often have
        # ties or near ties; because of that we can benefit by not resetting
        # the bounds unless we get stuck (don't join any components). Thus
        # we check for that, and only reset bounds in the case where we have
        # the same number of components as we did going in. This doesn't
        # produce a true min spanning tree, but only and approximation
        # Thus only do this if the caller is willing to accept such
        if self.approx_min_span_tree:
            last_num_components = self.components.shape[0]
            self.components = self.component_union_find.components()

            if self.components.shape[0] == last_num_components:
                # Reset bounds
                for n in range(self.num_nodes):
                    self.bounds[n] = np.inf
        else:
            self.components = self.component_union_find.components()

            for n in range(self.num_nodes):
                self.bounds[n] = np.inf

        return self.components.shape[0]

    def dual_tree_traversal(self, node1: np.int_, node2: np.int_):
        """Perform a dual tree traversal, pruning wherever possible, to find
        the nearest neighbor not in the same component for each component.
        This is akin to a standard dual tree NN search, but we also prune
        whenever all points in query and reference nodes are in the same
        component."""
        node1_info = self.node_data[node1]
        node2_info = self.node_data[node2]
        # Compute the distance between the query and reference nodes
        node_dist = KDTree.min_rdist_dual(self.tree, node1, self.tree, node2)

        # If the distance between the nodes is less than the current bound for
        # the query and the nodes are not in the same component continue;
        # otherwise we get to prune this branch and return early.
        if node_dist < self.bounds[node1]:
            if self.component_of_node[node1] == self.component_of_node[node2] and \
                    self.component_of_node[node1] >= 0:
                return
        else:
            return

        # Case 1: Both nodes are leaves
        #       for each pair of points in node1 x node2 we need
        #       to compute the distance and see if it better than
        #       the current nearest neighbor for the component of
        #       the point in the query node.
        #
        #       We get to take some shortcuts:
        #           - if the core distance for a point is larger than
        #             the distance to the nearst neighbor of the
        #             component of the point ... then we can't get
        #             a better mutual reachability distance and we
        #             can skip computing anything for that point
        #           - if the points are in the same component we
        #             don't have to compute the distance.
        #
        #       We also have some catches:
        #           - we need to compute mutual reachability distance
        #             not just the ordinary distance; this involves
        #             fiddling with core distances.
        #           - We need to scale distances according to alpha,
        #             but don't want to lose performance in the case
        #             that alpha is 1.0.
        #
        #       Finally we can compute new bounds for the query node
        #       based on the distances found here, so do that and
        #       propagate the results up the tree.
        if node1_info.is_leaf and node2_info.is_leaf:

            new_upper_bound = 0.0
            new_lower_bound = np.inf

            point_indices1 = self.idx_array[node1_info.idx_start:node1_info.idx_end]
            point_indices2 = self.idx_array[node2_info.idx_start:node2_info.idx_end]

            for i in range(point_indices1.shape[0]):
                p = point_indices1[i]
                component1 = self.component_of_point[p]

                if self.core_distance[p] > self.candidate_distance[component1]:
                    continue

                for j in range(point_indices2.shape[0]):
                    q = point_indices2[j]
                    component2 = self.component_of_point[q]

                    if self.core_distance[q] > self.candidate_distance[component1]:
                        continue

                    if component1 != component2:
                        d = self.clusterer.reduced_distance(self._raw_galaxies[p], self._raw_galaxies[q])

                        # mr_dist = max(distances[i, j],
                        #               self.core_distance_ptr[p],
                        #               self.core_distance_ptr[q])
                        if self.alpha != 1.0:
                            mr_dist = max(d / self.alpha, self.core_distance[p], self.core_distance[q])
                        else:
                            mr_dist = max(d, self.core_distance[p], self.core_distance[q])
                        if mr_dist < self.candidate_distance[component1]:
                            self.candidate_distance[component1] = mr_dist
                            self.candidate_neighbor[component1] = q
                            self.candidate_point[component1] = p

                new_upper_bound = max(new_upper_bound, self.candidate_distance[component1])
                new_lower_bound = min(new_lower_bound, self.candidate_distance[component1])

            # Compute new bounds for the query node, and
            # then propagate the results of that computation
            # up the tree.
            new_bound = min(new_upper_bound, new_lower_bound + 2 * node1_info.radius)
            # new_bound = new_upper_bound
            if new_bound < self.bounds[node1]:
                self.bounds[node1] = new_bound

                # Propagate bounds up the tree
                while node1 > 0:
                    parent = (node1 - 1) // 2
                    left = 2 * parent + 1
                    right = 2 * parent + 2

                    new_bound = max(self.bounds[left], self.bounds[right])

                    if new_bound < self.bounds[parent]:
                        self.bounds[parent] = new_bound
                        node1 = parent
                    else:
                        break

        # Case 2a: The query node is a leaf, or is smaller than
        #          the reference node.
        #
        #       We descend in the reference tree. We first
        #       compute distances between nodes to determine
        #       whether we should prioritise the left or
        #       right branch in the reference tree.
        elif node1_info.is_leaf or (not node2_info.is_leaf and node2_info.radius > node1_info.radius):

            left = 2 * node2 + 1
            right = 2 * node2 + 2

            left_dist = KDTree.min_rdist_dual(self.tree, node1, self.tree, left)

            right_dist = KDTree.min_rdist_dual(self.tree, node1, self.tree, right)

            if left_dist < right_dist:
                self.dual_tree_traversal(node1, left)
                self.dual_tree_traversal(node1, right)
            else:
                self.dual_tree_traversal(node1, right)
                self.dual_tree_traversal(node1, left)

        # Case 2b: The reference node is a leaf, or is smaller than
        #          the query node.
        #
        #       We descend in the query tree. We first
        #       compute distances between nodes to determine
        #       whether we should prioritise the left or
        #       right branch in the query tree.
        else:
            left = 2 * node1 + 1
            right = 2 * node1 + 2

            left_dist = KDTree.min_rdist_dual(self.tree, left, self.tree, node2)

            right_dist = KDTree.min_rdist_dual(self.tree, right, self.tree, node2)

            if left_dist < right_dist:
                self.dual_tree_traversal(left, node2)
                self.dual_tree_traversal(right, node2)
            else:
                self.dual_tree_traversal(right, node2)
                self.dual_tree_traversal(left, node2)

    def spanning_tree(self):
        """Compute the minimum spanning tree of the data held by the tree passed in at construction"""

        num_components = self.num_points
        while num_components > 1:
            self.dual_tree_traversal(0, 0)
            num_components = self.update_components()

        return self.edges
