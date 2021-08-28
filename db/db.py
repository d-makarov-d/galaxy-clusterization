from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict


class DataError(Exception):
    pass


class DBModel(ABC):
    """Entity for operating with specific collection"""
    def __init__(self, database: Database):
        self._collection: Collection = database[self.collection_name]
        # ensure collection has the right validator
        validator = {'$jsonSchema': self.schema}
        database.command(OrderedDict([
            ('collMod', self.collection_name),
            ('validator', validator)
        ]))

    @property
    @abstractmethod
    def schema(self) -> dict:
        """JSON schema"""
        pass

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Collection name"""
        pass

    @property
    @abstractmethod
    def instance_class(self) -> DBInstance.__class__:
        pass

    @property
    def collection(self):
        return self._collection

    def save(self, descr: dict):
        """
        Creates new database entry
        :param descr: Entry description
        """
        self._collection.insert_one(descr)

    def save_instance(self, instance: DBInstance):
        self.save(instance.to_dict())

    def find(self, query: dict) -> list[DBInstance]:
        return list(map(
            lambda el: self.instance_class.from_db(el),
            self._collection.find(query)
        ))

    def find_one(self, query: dict) -> DBInstance:
        descr = self._collection.find_one(query)
        if descr:
            return self.instance_class.from_db(descr)
        else:
            return None


class DBInstance(ABC):
    """Represents collection entry"""
    @staticmethod
    @abstractmethod
    def from_db(data: dict) -> DBInstance:
        """
        Initialize Object from database data
        :param data: Result from database query
        :return: Created object
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Return dict to save to db"""
        pass