from __future__ import annotations

from .db import DBInstance, DataError
from cosmological import CosmologicalParams


class Galaxy(DBInstance):
    def __init__(self, dist: float, ra: float, dec: float, mass: float, ev: float):
        """
        Describes a galaxy instance. All parameters are expected in absolute values, not relative to instrument
        :param dist: Distance [Mpc]
        :param ra: Right extension (aka longitude) [rad]
        :param dec: Declination (aka latitude) [rad]
        :param mass: Stellar mass corrected
        :param ev: Velocity calculation error [km/sec]
        """
        self._dist = dist
        self._ra = ra
        self._dec = dec
        self._mass = mass
        self._ev = ev

    @property
    def dist(self) -> float:
        return self._dist

    @property
    def ra(self) -> float:
        return self._ra

    @property
    def dec(self) -> float:
        return self._dec

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def ev(self) -> float:
        return self._ev

    @staticmethod
    def from_db(data: dict, params: CosmologicalParams = None) -> Galaxy:
        """Unpack dictionary, returned by database, to pythonic structure"""
        if params is None:
            params = CosmologicalParams()
        # velocity relative to Local group [km/sec]
        vlg = data.get('vlg')
        if vlg is None:
            raise DataError('vlg field missing')
        # vlg error TODO: mb dist error
        ev = data.get('ev')
        if ev is None:
            raise DataError('ev field missing')
        # TODO: warning
        # Stellar magnitude corrected
        mag = data.get('mag') or params.gal_mag_placeholder
        # Right extension (aka longitude)
        ra = data.get('al') or data.get('al2000') or data.get('ra')
        if ra is None:
            raise DataError('[al | al2000 | ra] field missing')
        # Declination (aka latitude)
        dec = data.get('de') or data.get('de2000') or data.get('dec')
        if dec is None:
            raise DataError('[de | de2000 | dec] field missing')
        dist = vlg / params.H0
        return Galaxy(dist, ra, dec, mag, ev)

    def to_dict(self) -> dict:
        pass
