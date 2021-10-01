from __future__ import annotations

from astropy.coordinates import spherical_to_cartesian
import uuid

from ._db_abc import DBModel, DBInstance, DBError, TableDesr
from cosmological import CosmologicalParams


class Galaxy(DBInstance):
    def __init__(self, dist: float, ra: float, dec: float, mass: float, ed: float, _id: str = None):
        """
        Describes a galaxy instance. All parameters are expected in absolute values, not relative to instrument
        :param dist: Distance [Mpc]
        :param ra: Right extension (aka longitude) [rad]
        :param dec: Declination (aka latitude) [rad]
        :param mass: Stellar mass corrected
        :param ed: Distance calculation error [Mpc]
        """
        self._dist = dist
        self._ra = ra
        self._dec = dec
        self._mass = mass
        self._ed = ed
        if _id is None:
            self._id = str(uuid.uuid4())
        else:
            self._id = _id

        x, y, z = spherical_to_cartesian(self.dist, self.dec, self.ra)
        self._x = x.value
        self._y = y.value
        self._z = z.value

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
    def ed(self) -> float:
        return self._ed

    @property
    def cart(self) -> tuple[float, float, float]:
        """Cartesian coordinates, [X, Y, Z]"""
        return self._x, self._y, self._z

    @property
    def split_coordinates(self) -> [float]:
        """Returns spatial coordinates, by which BallTree and KDTree will split the space"""
        return self.cart

    @staticmethod
    def from_db(data: tuple, params: CosmologicalParams = None) -> Galaxy:
        return Galaxy(*data[1:], data[0])

    def to_tuple(self) -> tuple:
        return (
            self._id,
            self.dist,
            self.ra,
            self.dec,
            self.mass,
            self.ed
        )

    @staticmethod
    def table_descr() -> TableDesr:
        fields = [
            TableDesr.Field('dist', float),
            TableDesr.Field('right_extension', float),
            TableDesr.Field('declination', float),
            TableDesr.Field('stellar_mass', float),
            TableDesr.Field('dist_error', float),
        ]
        _id = TableDesr.Field('name', str)
        return TableDesr('galaxies', fields, _id)


class GalaxiesDB(DBModel):
    @property
    def schema(self) -> tuple[TableDesr]:
        return Galaxy.table_descr(),
