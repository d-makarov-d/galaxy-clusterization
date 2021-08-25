from __future__ import annotations

from .db import DBModel, DBInstance


class Galaxy(DBInstance):
    def __init__(self, z: float, alpha: float, delta: float, lum: float):
        """
        Describes a galaxy instance
        :param z: Redshift
        :param alpha: Right extension
        :param delta: Declination
        :param lum: Luminosity
        """
        self._z = z
        self._alpha = alpha
        self._delta = delta
        self._lum = lum

    @property
    def z(self) -> float:
        return self._z

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def lum(self) -> float:
        return self._lum

    @staticmethod
    def from_db(data: dict) -> Galaxy:
        pass

    def to_dict(self) -> dict:
        pass
