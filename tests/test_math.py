import unittest
import numpy as np
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

from db.galaxy import Galaxy
from cosmological_cluster import CosmologicalCluster
from algorithms import nts_element


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

    def test_nth_element(self):
        """Nth element should behave correctly"""
        def test(data, idx_start, idx_end):
            idx_arr = np.arange(len(data))
            prev = list(data[idx_arr])
            nts_element(data, idx_arr, idx_start, idx_end)
            got = list(data[idx_arr])
            self.assertListEqual(prev[:idx_start], got[:idx_start])
            self.assertListEqual(prev[idx_end:], got[idx_end:])
            idx_mid = int((idx_end - idx_start) / 2)
            self.assertEqual(got[idx_start:idx_end][idx_mid], np.sort(data[idx_start:idx_end])[idx_mid])
            left_part = got[idx_start:idx_start + idx_mid]
            right_part = got[idx_start + idx_mid+1:idx_end]
            self.assertEqual(0, sum(map(lambda el: el > got[idx_start + idx_mid], left_part)),
                             "All left-hand parts of nth_elements result should be less than middle")
            self.assertEqual(0, sum(map(lambda el: el < got[idx_start + idx_mid], right_part)),
                             "All right-hand parts of nth_elements result should be greater than middle")

        test(np.linspace(9, 0, 10), 2, 8)
        test(np.linspace(1, 0, 11), 2, 9)
        data = np.linspace(0, 1, 100)
        np.random.shuffle(data)
        test(data, 5, 15)
