from typing import Iterable, Tuple
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
import numpy as np
from numpy.typing import NDArray

from clusterization import Cluster


class CosmologicalCluster(Cluster):
    def merge(self,
              dist: NDArray, ra: NDArray, dec: NDArray, mass: NDArray, ev: NDArray
              ) -> Tuple[float, float, float, float, float]:
        cartesian = np.transpose(np.array(spherical_to_cartesian(dist, dec, ra)))
        center_cart = np.average(cartesian, axis=0, weights=mass)
        (_, cent_lat, cent_lon) = cartesian_to_spherical(*center_cart)  # r, lat, lon
        return (
            np.average(dist, weights=1/ev),     # dist
            cent_lon,                           # ra
            cent_lat,                           # dec
            np.sum(mass),                       # mass
            max(np.std(dist), np.mean(ev)) / np.sqrt(len(dist))  # ev, TODO check formula
        )
