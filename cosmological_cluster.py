from typing import Iterable, Tuple
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
import numpy as np

from clusterization import Cluster
from db.galaxy import Galaxy


class CosmologicalCluster(Cluster):
    def merge(self, members: tuple[Galaxy]) -> Tuple[float, float, float, float, float]:
        dist = np.zeros(len(members))
        ra = np.zeros(len(members))
        dec = np.zeros(len(members))
        mass = np.zeros(len(members))
        ed = np.zeros(len(members))
        cartesian = np.zeros((3, len(members)))
        for i in range(0, len(members)):
            dist[i] = members[i].dist
            ra[i] = members[i].ra
            dec[i] = members[i].dec
            mass[i] = members[i].mass
            ed[i] = members[i].ed
            cartesian[:, i] = members[i].cart
        center_cart = np.average(cartesian, axis=1, weights=mass)
        (_, cent_lat, cent_lon) = cartesian_to_spherical(*center_cart)  # r, lat, lon
        return (
            np.average(dist, weights=1 / ed),  # dist
            cent_lon,  # ra
            cent_lat,  # dec
            np.sum(mass),  # mass
            max(np.std(dist), np.mean(ed)) / np.sqrt(len(dist))  # ev, TODO check formula
        )
