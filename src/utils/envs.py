"""
Simple wrapper for work with environmental variables within repository setting.

Usage can be found in logger and config files and in jupyter notebook
/notebooks/documentation/low_level_tools_documentation.py or
/docs/ low_level_tools_documentation.html.
"""
from os import environ

from src.constants.global_constants import ENV_CONFIG, DEFAULT_CONFIG, ENV_LOGGER, DEFAULT_LOGGER, \
    ENV_RUNNING_UNIT_TESTS, DEFAULT_RUNNING_UNIT_TESTS


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
        :param value: str. Value to be set without .conf.
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
        :param value: str. Value to be set without .conf.
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

    @staticmethod
    def set_running_unit_tests(value: str = "True") -> None:
        """
        Sets running unit tests.
        """
        environ[ENV_RUNNING_UNIT_TESTS] = value

    @staticmethod
    def get_running_unit_tests() -> str:
        """
        Returns if currently unit tests are running.
        :return: bool. If running or not
        """
        value = environ.get(ENV_RUNNING_UNIT_TESTS)
        if value is None:
            value = DEFAULT_RUNNING_UNIT_TESTS
        return value
