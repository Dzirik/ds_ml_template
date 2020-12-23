"""
Code for automatic parameterized notebook execution.

Based on papermill library.
"""
import os
from typing import List, Dict, Union

import papermill

from src.utils.config import Config

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

    def __init__(self) -> None:
        self.config = Config()

        self.ntb_path: str
        self.output_folder: str
        self.list_of_params: List[Dict[str, Union[str, float]]]

    def _set_up_params(self) -> None:
        """
        Sets up the params for run. If there is specified in config not to do it, it returns default values,
        otherwise it gets param from config.
        """
        if self.config.get().param_ntb_execution.use_default:
            self.ntb_path = DEFAULT_NTB_PATH
            self.output_folder = DEFAULT_OUTPUT_FOLDER
            self.list_of_params = DEFAULT_LIST_OF_PARAMS
        else:
            self.ntb_path = self.config.get().param_ntb_execution.ntb_path
            self.output_folder = self.config.get().param_ntb_execution.output_folder
            self.list_of_params = self.config.get().param_ntb_execution.notebook_executioner_params

    def execute(self, convert_to_html: bool = True) -> None:
        """
        Executes the notebook based on default params or config params.
        :param convert_to_html: bool. If to convert to html or not.
        """
        self._set_up_params()
        n = 0
        for params in self.list_of_params:
            n = n + 1
            path_out = os.path.abspath(os.path.join(self.output_folder, "notebook_" + str(n) + ".ipynb"))
            papermill.execute_notebook(self.ntb_path, path_out, params)
            if convert_to_html:
                os.system("jupyter nbconvert --to html " + path_out)


if __name__ == "__main__":
    EXECUTIONER = ParamNotebookExecutioner()
    EXECUTIONER.execute()
