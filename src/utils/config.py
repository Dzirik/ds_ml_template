"""
Code for handling configurations through the repository.

A singleton pattern is used for that.

Usage can be found in the end of the file and in jupyter notebook
/notebooks/documentation/low_level_tools_documentation.py or
/docs/ low_level_tools_documentation.html.
"""

from os import getcwd
from os.path import join, exists
from typing import NamedTuple, List, Dict, Union

import typedload
from pyhocon import ConfigFactory

from src.constants.global_constants import FOLDER_CONFIGURATIONS
from src.utils.envs import Envs
from src.utils.logger import Logger
from src.utils.singleton_meta import Singleton


class Path(NamedTuple):
    """
    Configuration tuple for paths to data files.
    """
    archive_data: str
    external_data: str
    interim_data: str
    processed_data: str
    raw_data: str
    results_data: str


class DatabaseCredentials(NamedTuple):
    """
    Log in credentials for data base.
    """
    provider: str
    user: str
    password: str
    host: str
    port: int
    db_name: str


class DashSettings(NamedTuple):
    """
    Configuration tuple for the dash setting.
    """
    header_link_color: str
    header_font_weight: str
    navbar_background_color: str
    path_to_image: str


class Dash(NamedTuple):
    """
    Configuration tuple for the dash.
    """
    sidebar_config: List[List[List[str]]]
    list_of_pages: Dict[str, List[str]]
    sidebar_style: Dict[str, Union[int, str]]
    content_style: Dict[str, str]
    sett: DashSettings


class ParamNotebookExecution(NamedTuple):
    """
    Configuration tuple for the parameterized notebook execution.
    """
    use_default: bool
    convert_to_html: bool
    ntb_path: str
    output_folder: str
    notebook_executioner_params: List[Dict[str, Union[float, str]]]


class Profile(NamedTuple):
    """
    Overall configuration tuple for everything.
    """
    name: str
    path: Path
    db_cred: DatabaseCredentials
    dash: Dash
    param_ntb_execution: ParamNotebookExecution


class Config(metaclass=Singleton):
    """
    Class for storing or configuration options for the repository.

    Singleton class.

    Takes settings from environmental variables or uses default "python_local.conf".
    """
    _is_profile = False
    _profile: Profile
    _env = Envs()

    def __init__(self) -> None:
        if not self._is_profile:
            self._profile = ConfigFactory.parse_file(self._get_profile_file_path(self._env.get_config()))
            self._profile = typedload.load(self._profile, Profile)

            Logger().debug(f"Python config was created from {self._env.get_config()}.conf file.")

            self._is_profile = True

    def get(self) -> Profile:
        """
        Gets the _profile as Profile NamedTuple class.
        :return: Profile. NamedTuple containing settings.
        """
        return self._profile

    @staticmethod
    def _get_profile_file_path(profile_name: str) -> str:
        """
        Reads configuration file of name profile_name from configuration folder. Raises an exception if there is no
        file.
        :param profile_name: Optional[str]. Name of the _profile and the file at the same time. File name without .conf.
        """
        profile_file_name = f"{profile_name}.conf"
        profile_file_path = join("../../", FOLDER_CONFIGURATIONS, profile_file_name)

        if not exists(profile_file_path):
            # because of problems with running dash drom /index.py - in logger as well
            profile_file_path = join(getcwd(), FOLDER_CONFIGURATIONS, profile_file_name)

            if not exists(profile_file_path):
                Logger().error("Logger profile does not exist in the selected path.")
                raise ValueError("Config profile does not exist in the selected path.")

        return profile_file_path


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
    print(c.get().name)
    print(c.get().path.external_data)

    print(c.get().dash.sidebar_config)

    print(len(c.get().dash.sidebar_config[1]))
    print(len(c.get().dash))
    for item in c.get().dash:
        print(item)
