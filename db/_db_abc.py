from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Collection, Union
import sqlite3


class DBError(Exception):
    pass


class DBModel(ABC):
    """Entity for operating with database in a specific file"""
    def __init__(self, filename: Union[str, sqlite3.Connection]):
        if isinstance(filename, str):
            self.con = sqlite3.connect(filename)
        else:
            self.con = filename
        # ensure, that db contains right tables
        expected_names = tuple(map(lambda el: el.name, self.schema))
        cursor = self.con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        names = list(map(lambda el: el[0], cursor.fetchall()))
        for name in names:
            if name not in expected_names:
                self.close()
                raise DBError('Unexpected table "%s"' % name)
        for i, name in enumerate(expected_names):
            if name not in names:
                cursor.execute(str(self.schema[i]))

    @property
    @abstractmethod
    def schema(self) -> tuple[TableDesr]:
        """Describes table fields"""
        pass

    def save(self, data: Union[DBInstance, Iterable[DBInstance]]):
        if isinstance(data, Iterable):
            to_insert = list(map(lambda el: el.to_tuple(), data))
            descr = data.__iter__().__next__().table_descr()
            self.con.executemany(descr.insert_str, to_insert)
        else:
            self.con.execute(data.table_descr().insert_str, data.to_tuple())

    def find(self, item_t: DBInstance.__class__, query: str = None) -> list[DBInstance]:
        descr = item_t.table_descr()
        if descr.name not in tuple(map(lambda el: el.name, self.schema)):
            raise DBError('This DB does not have table for this instances')

        if query is None:
            cursor = self.con.cursor().execute('SELECT * FROM %s' % descr.name)
        else:
            # TODO check query value!!!
            cursor = self.con.cursor().execute('SELECT * FROM %s WHERE %s' % (descr.name, query))

        return tuple(map(lambda el: item_t.from_db(el), cursor.fetchall()))


    def find_one(self, query: dict) -> DBInstance:
        # TODO
        pass

    def close(self):
        self.con.close()


class TableDesr:
    """Describes table row fields"""
    def __init__(self, name: str, items: Collection[TableDesr.Field], _id: Union[None, TableDesr.Field]):
        self.name = name
        self.items = items
        self._id = _id

    def __str__(self):
        if self._id is None:
            fields = ''
        else:
            fields = f'\t{self._id.name} {self._id.type} PRIMARY KEY'
        for item in self.items:
            if len(fields) > 0:
                fields += ',\n '
            fields += f'\t{item.name} {item.type}{item.required}{item.unique}'

        return 'CREATE TABLE %s(\n%s\n);' % (self.name, fields)

    @property
    def insert_str(self) -> str:
        n_values = len(self.items)
        if self._id is not None:
            n_values += 1
        return 'INSERT INTO %s VALUES(%s);' % (self.name, (n_values * '?,')[:-1])

    class Field:
        """Table row field description"""
        def __init__(self, name: str, py_type, required=False, unique=False):
            if py_type not in self.py_to_sqlite_types().keys():
                raise DBError('Unsupported type: %s for field %s' % (py_type, name))
            self.name = name
            self.type = self.py_to_sqlite_types()[py_type]
            self.required = ''
            self.unique = ''
            if required:
                self.required = ' NOT NULL'
            if unique:
                self.unique = ' UNIQUE'

        @staticmethod
        def py_to_sqlite_types() -> dict:
            return {
                None: 'NULL',
                int: 'INTEGER',
                float: 'REAL',
                str: 'TEXT',
                bytes: 'BLOB',
            }


class DBInstance(ABC):
    """Represents table row entry"""
    @staticmethod
    @abstractmethod
    def table_descr() -> TableDesr:
        pass

    @staticmethod
    @abstractmethod
    def from_db(data: tuple) -> DBInstance:
        """
        Initialize Object from database data
        TODO find another solution
        IMPORTANT!!! ID must go first, and other values in same order as in table_descr
        :param data: Result from database query
        :return: Created object
        """
        pass

    @abstractmethod
    def to_tuple(self) -> tuple:
        """Return dict to save to db.
        TODO find another solution
        IMPORTANT!!! ID must go first, and other values in same order as in table_descr"""
        pass
