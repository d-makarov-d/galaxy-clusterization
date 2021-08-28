from astropy import constants
from astropy.cosmology import WMAP9 as cosmo

default_gal_mag_placeholder = 16.5
default_ml = 6
default_H0 = cosmo.H(0).value
default_mass_cnt = 1e11 * constants.pc.value * constants.G.value * constants.M_sun.value


class CosmologicalParams:
    def __init__(self,
                 gal_mag_placeholder: float=default_gal_mag_placeholder,
                 ml: float=default_ml,
                 H0: float=default_H0,
                 mass_cnt=default_mass_cnt
                 ):
        """
        :param gal_mag_placeholder: Placeholder for galaxies with missing magnitude [???]
        :param ml: Mass to luminosity ration [???]
        :param H0: Hubble's constant [km / (Mpc s)]
        :param mass_cnt: ???
        """
        self._gal_mag_placeholder = gal_mag_placeholder
        self._ml = ml
        self._H0 = H0
        self._mass_cnt = mass_cnt

    @property
    def gal_mag_placeholder(self) -> float:
        """:return Placeholder for galaxies with missing magnitude [???]"""
        return self._gal_mag_placeholder

    @property
    def ml(self) -> float:
        """:return Mass to luminosity ration [???]"""
        return self._ml

    @property
    def H0(self) -> float:
        """:return Hubble's constant [km / (Mpc s)]"""
        return self._H0

    @property
    def mass_cnt(self) -> float:
        """:return ???"""
        return self._mass_cnt
