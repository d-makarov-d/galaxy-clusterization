from typing import Collection
import numpy as np
from astropy.coordinates import spherical_to_cartesian

from db.galaxy import Galaxy
from clusterization import Cluster

cluster_colors = ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000', '#029386', '#f97306']


def draw_galaxies(ax, g: Collection[Galaxy], c=None, **kwargs):
    """Draws a galaxy set with errors, circle size corresponds to galaxy mass"""
    x, y, z, m = (np.empty(len(g)), np.empty(len(g)), np.empty(len(g)), np.empty(len(g)))
    err_x, err_y, err_z = (np.zeros(len(g) * 3), np.zeros(len(g) * 3), np.zeros(len(g) * 3))
    for i, galaxy in enumerate(g):
        x[i], y[i], z[i] = galaxy.cart
        err_x[3*i], err_y[3*i], err_z[3*i] = np.nan, np.nan, np.nan
        err_x[3*i + 1], err_y[3*i + 1], err_z[3*i + 1] = \
            spherical_to_cartesian(galaxy.dist - galaxy.ed / 2., galaxy.dec, galaxy.ra)
        err_x[3*i + 2], err_y[3*i + 2], err_z[3*i + 2] = \
            spherical_to_cartesian(galaxy.dist + galaxy.ed / 2., galaxy.dec, galaxy.ra)
        m[i] = galaxy.mass * 10

    ax.scatter(x, y, z, s=np.round(m), c=c, **kwargs)
    ax.plot(err_x, err_y, err_z, c='r', alpha=0.4)


def draw_clusters(ax, clusters: Collection[Cluster]):
    available_colors = list(cluster_colors)
    colors = []
    from_clustered = []
    for cluster in clusters:
        if isinstance(cluster, Cluster):
            from_clustered += cluster.galaxies
            n_gals = len(cluster.galaxies)
            if n_gals > 3:
                colors += n_gals * [available_colors.pop()]
            else:
                colors += n_gals * ['k']
        else:
            colors.append('k')
            from_clustered.append(cluster)
    draw_galaxies(ax, from_clustered, c=colors)
