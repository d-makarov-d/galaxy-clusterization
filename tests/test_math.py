import unittest
import numpy as np
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

from db.galaxy import Galaxy
from cosmological_cluster import CosmologicalCluster


class TestMath(unittest.TestCase):
    def test_sphere_center(self):
        """Test if finding a center of in cartesian coordinates is equivalent to sphere system"""
        n_points = 40
        points = (np.random.rand(n_points, 3) - 0.5) * 6 + 1
        magnitudes = np.random.rand(n_points)
        center_cart = np.average(points, axis=0, weights=magnitudes)

        def point_to_galaxy(p: tuple[float]):
            (r, lat, lon) = cartesian_to_spherical(*p[:3])
            return Galaxy(r.value, lon.value, lat.value, p[3], p[3])
        galaxies = tuple(map(point_to_galaxy, np.concatenate((points, np.expand_dims(magnitudes, axis=1)), axis=1)))
        cluster = CosmologicalCluster(galaxies)

        sphere_center = (cartesian_to_spherical(*center_cart)[0], cluster.dec, cluster.ra)  # r, lat, lon
        cluster_center = np.array(spherical_to_cartesian(*sphere_center))
        dist = np.sqrt(np.sum((center_cart - cluster_center)**2))
        self.assertTrue(dist < 1e-15)
