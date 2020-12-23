"""
Simple wrapper for work with environmental variables within repository setting.
"""
from os import environ

from src.global_constants import ENV_CONFIG, DEFAULT_CONFIG, ENV_LOGGER, DEFAULT_LOGGER


class Envs:
    """
    Class for handling environmental variables for configuration.

    It tried to get one from environment, if it cannot get it, it uses the default configuration option.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def set_config(value: str) -> None:
        """
        Sets the environmental variable for config singleton with value.
        :param name: str. Value to be setted.
        :return:
        """
        environ[ENV_CONFIG] = value

    @staticmethod
    def get_config() -> str:
        """
        Returns the value of environmental variable for config singleton or none.
        :return: Optional[str]. Value or none.
        """
        value = environ.get(ENV_CONFIG)
        if value is None:
            value = DEFAULT_CONFIG
        return value

    @staticmethod
    def set_logger(value: str) -> None:
        """
        Sets the environmental variable for logger singleton with value.
        :param name: str. Value to be setted.
        :return:
        """
        environ[ENV_LOGGER] = value

    @staticmethod
    def get_logger() -> str:
        """
        Returns the value of environmental variable for logger singleton or none.
        :return: Optional[str]. Value or none.
        """
        value = environ.get(ENV_LOGGER)
        if value is None:
            value = DEFAULT_LOGGER
        return value
