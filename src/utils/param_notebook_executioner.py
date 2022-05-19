"""
Code for automatic parameterized notebook execution.

Based on papermill library.

IMPORTANT:
- The notebook has to be in *.ipynb instead of *.py.
- The script works being run from PyCharm. There is a problem with paths running it from console.
"""
import os
from typing import List, Dict, Union, Optional

import papermill

from src.utils.config import Config
from src.utils.envs import Envs
from src.utils.timer import Timer

# THESE PARAMETERS ARE IN CONFIG FILE
DEFAULT_NTB_PATH = "../../notebooks/template/template_parameterized_execution_notebook.ipynb"
DEFAULT_OUTPUT_FOLDER = "../../reports"
DEFAULT_LIST_OF_PARAMS: List[Dict[str, Union[str, float]]] = [
    {"n": 10, "a": 1, "b": 1, "title": "Positive"},
    {"n": 15, "a": -1, "b": -1, "title": "Negative"},
    {"n": 20, "a": 0, "b": 2, "title": "Zero"}
]


class ParamNotebookExecutioner:
    """
    Class for execution of the parameterized notebook with different set of parameters for each run.
    """

    def __init__(self, config_name: Optional[str] = None) -> None:
        """
        :param config_name: Optional[str]. If None, uses default one. Otherwise, using the config_name one.
        """
        if config_name is not None:
            envs = Envs()
            envs.set_config(config_name)
        self._config = Config()

        self._ntb_path: str
        self._output_folder: str
        self._list_of_params: List[Dict[str, Union[str, float]]]

    def _set_up_params(self) -> None:
        """
        Sets up the params for run. If there is specified in config not to do it, it returns default values,
        otherwise it gets param from config.
        """
        if self._config.get().param_ntb_execution.use_default:
            self._ntb_path = DEFAULT_NTB_PATH
            self._output_folder = DEFAULT_OUTPUT_FOLDER
            self._list_of_params = DEFAULT_LIST_OF_PARAMS
        else:
            self._ntb_path = self._config.get().param_ntb_execution.ntb_path
            self._output_folder = self._config.get().param_ntb_execution.output_folder
            self._list_of_params = self._config.get().param_ntb_execution.notebook_executioner_params

    def execute(self, notebook_name: str = "notebook_", name_with_number: bool = False, convert_to_html: bool = True) \
            -> None:
        """
        Executes the notebook based on default params or config params.
        :param notebook_name: str. First part of notebook name.
        :param name_with_number: bool. If True, then notebooks are named with number based on the order of execution.
                                       If False, then the notebook is named based on parameters converted to string.
        :param convert_to_html: bool. If to convert to html or not.
        """
        self._set_up_params()
        n = 0
        for params in self._list_of_params:
            if name_with_number:
                n = n + 1
                name = str(n)
            else:
                name = ""
                for _, value in params.items():
                    name = name + str(value) + "_"
            path_out = os.path.abspath(os.path.join(self._output_folder, notebook_name + str(name) + ".ipynb"))
            papermill.execute_notebook(self._ntb_path, path_out, params)
            if convert_to_html:
                os.system("jupyter nbconvert --to html " + path_out)


if __name__ == "__main__":
    TIMER = Timer()

    CONFIG_NAME = None  # None
    NOTEBOOK_NAME = "notebook_"
    NAME_WITH_NUMBER = False

    TIMER.start()
    EXECUTIONER = ParamNotebookExecutioner(CONFIG_NAME)
    EXECUTIONER.execute(notebook_name=NOTEBOOK_NAME, name_with_number=NAME_WITH_NUMBER)
    TIMER.end(label="End of Notebook Executioner")
