"""Generates artificial galaxies distribution with groups"""
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from db.galaxy import GalaxiesDB

from db.galaxy import Galaxy
from visualization import draw_galaxies

H = cosmo.H(0).value


class Pos:
    @staticmethod
    def from_cart(x, y, z):
        r, lat, lon = cartesian_to_spherical(x, y, z)
        return Pos(x, y, z, r.value, lat.value, lon.value)

    @staticmethod
    def from_sph(r, lat, lon):
        x, y, z = spherical_to_cartesian(r, lat, lon)
        return Pos(x.value, y.value, z.value, r, lat, lon)

    def __init__(self, x, y, z, r, lat, lon):
        self.x, self.y, self.z = x, y, z
        self.r, self.lat, self.lon = r, lat, lon


def mk_group(pos: Pos, n_members: int, M: float, R: float, dV: float) -> list[Galaxy]:
    """
    Generates a group of galaxies
    :param pos: Group center galaxy position
    :param n_members: Number of group members
    :param M: Group mass
    :param R: Group size
    :param dV: Velocity dispersion
    :return: List of galaxies
    """
    x = np.random.normal(pos.x, R, n_members)
    y = np.random.normal(pos.y, R, n_members)
    z = np.random.normal(pos.z, R, n_members)
    ed = np.ones(n_members) * dV / H
    m = [np.random.normal(2 * M / n_members, 1 / n_members)]
    for i in range(0, n_members - 2):
        m.append(np.random.normal((M * (n_members - 2) / n_members) / (n_members - 1), 1 / n_members))
    m.append(M - sum(m))
    galaxies = []
    for i in range(0, n_members):
        r, lat, lon = cartesian_to_spherical(x[i], y[i], z[i])
        galaxies.append(Galaxy(r.value, lon.value, lat.value, m[i], ed[i]))
    return galaxies


if __name__ == '__main__':
    gals = []
    # generate groups
    for i in range(0, 3):
        lat = np.pi * (np.random.random() - 0.5)
        lon = 2 * np.pi * np.random.random()
        gals += mk_group(Pos.from_sph(3.7, lat, lon), 31, 12.49, 0.219, 123)
    # random noise
    max_r = max(list(map(lambda g: g.dist, gals))) * 2.1
    for i in range(0, 100):
        x = (np.random.random() - 0.5) * 2 * max_r
        y = (np.random.random() - 0.5) * 2 * max_r
        z = (np.random.random() - 0.5) * 2 * max_r
        r, lat, lon = cartesian_to_spherical(x, y, z)
        m = np.random.random() * 1.5
        ed = np.random.random() * 150 / H
        gals.append(Galaxy(r.value, lon.value, lat.value, m, ed))

    db = GalaxiesDB('test_groups.db')
    db.save(gals)
