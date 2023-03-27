"""
Code for handling configurations through the repository.

The interface should be the same as in other configs, inheritance is not possible - cyrcular dependency.

A singleton pattern is used for that.

Usage can be found in the end of the file and in jupyter notebook
/notebooks/documentation/low_level_tools_documentation.py or
/docs/ low_level_tools_documentation.html.
"""
from os import getcwd
from os.path import join, exists
from typing import Dict, Any

import typedload
from pyhocon import ConfigFactory

from src.constants.global_constants import FOLDER_CONFIGURATIONS
from src.utils.config_data import ConfigData
from src.utils.envs import Envs
from src.utils.logger import Logger
from src.utils.singleton_meta import Singleton


def _get_config_file_path(config_file_name: str) -> str:
    """
    Reads configuration file of name profile_name from configuration folder. Raises an exception if there is no
    file.
    :param config_file_name: Optional[str]. Name of the _profile and the file at the same time. File name without .conf.
    """
    profile_file_name = f"{config_file_name}.conf"
    profile_file_path = join("../../", FOLDER_CONFIGURATIONS, profile_file_name)

    if not exists(profile_file_path):
        # because of problems with running dash drom /index.py - in logger as well
        profile_file_path = join(getcwd(), FOLDER_CONFIGURATIONS, profile_file_name)

        if not exists(profile_file_path):
            Logger().error("Logger profile does not exist in the selected path.")
            raise ValueError("Config profile does not exist in the selected path.")

    return profile_file_path


class Config(metaclass=Singleton):
    """
    Class for storing or configuration options for the repository.

    Singleton class.

    Takes settings from environmental variables or uses default "python_local.conf".

    Methods should the same as in src/utils/base_config.py
    """

    def __init__(self) -> None:
        self._data: ConfigData
        self._is_profile = False
        self._env = Envs()

        self._parse_config()

    def _parse_config(self) -> None:
        """
        Parses the config.
        """
        if not self._is_profile:
            self._data = ConfigFactory.parse_file(_get_config_file_path(self._env.get_config()))
            self._data = typedload.load(self._data, ConfigData)

            Logger().debug(f"Python config was created from {self._env.get_config()}.conf file.")

            self._is_profile = True

    def get_data(self) -> ConfigData:
        """
        Gets the _profile as Profile NamedTuple class.
        :return: Profile. NamedTuple containing settings.
        """
        return self._data

    def get_data_as_dict(self) -> Dict[Any, Any]:
        """
        Gets the data (named tuple) into dictionary.
        :return: Dict[Any, Any].
        """
        return dict(self._data._asdict())


if __name__ == "__main__":
    # change this variable if you want to test default
    USE_DEFAULT = True
    NONE_DEFAULT_PYTHON_CONFIG_NAME = "python_local"

    if not USE_DEFAULT:
        env = Envs()
        env.set_config(NONE_DEFAULT_PYTHON_CONFIG_NAME)

    config_1 = Config()
    config_2 = Config()
    c = Config()

    print("\n")
    print(f"Is it just one object? {config_1 is config_2}")

    print("\n")
    print(c.get_data().name)
    print(c.get_data().path.external_data)
    print(c.get_data().path.logs_data)
    print(c.get_data().path.models)

    print(c.get_data().dash.sidebar_config)

    print(len(c.get_data().dash.sidebar_config[1]))
    print(len(c.get_data().dash))
    for item in c.get_data().dash:
        print(item)
