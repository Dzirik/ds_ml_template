"""
Connector to PostgreSQL database.

A facade pattern is used for that.
"""

from abc import ABCMeta, abstractmethod
from inspect import getmembers, isclass, isabstract
from sys import modules
from typing import Any

from sqlalchemy import create_engine

from src.utils.config import Config


class AbstractDBConnector(metaclass=ABCMeta):
    """
    Abstract facade for connection.
    """

    @abstractmethod
    def get_connection(self) -> Any:
        """
        Returns connection do the database.
        :return: Any.
        """


class PostgreDBConnector(AbstractDBConnector):
    """
    Connector for PostgreSQL data base.
    """

    def __init__(self) -> None:
        self.config = Config()

    def get_connection(self) -> Any:
        """
        Returns connection do the database.
        :return: Any. Connection to database.
        """
        return create_engine(self._create_connection_string())

    def _create_connection_string(self) -> str:
        connection_string = f"postgresql+psycopg2://" \
                            f"{self.config.get().db_cred.user}:" \
                            f"{self.config.get().db_cred.password}@" \
                            f"{self.config.get().db_cred.host}/" \
                            f"{self.config.get().db_cred.db_name}"
        return connection_string


class DBConnectorFactory:
    """
    Class for handling all connectors.
    """

    def __init__(self) -> None:
        self.config = Config()

    def create(self) -> Any:
        """
        Creates and returns proper connector.
        :return: Connector class.
        """
        classes = getmembers(
            modules[__name__],
            lambda c: (isclass(c) and not isabstract(c) and issubclass(c, AbstractDBConnector))
        )
        for class_name, class_type in classes:
            if class_name == self.config.get().db_cred.provider:
                return class_type()
        raise Exception("There is no such connection.")
