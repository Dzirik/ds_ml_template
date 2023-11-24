"""
Code for automatic parameterized notebook execution.

Based on papermill library.

IMPORTANT:
- The notebook has to be in *.ipynb instead of *.py.
- The script works being run from PyCharm. There is a problem with paths running it from console.
"""
import os
from datetime import datetime
from typing import List, Dict, Union, Optional

import papermill

from src.utils.config import Config
from src.utils.date_time_functions import create_datetime_id
from src.utils.envs import Envs
from src.utils.timer import Timer

# THESE PARAMETERS ARE IN CONFIG FILE
DEFAULT_NTB_PATH = "../../notebooks/template/template_parameterized_execution_notebook.ipynb"
DEFAULT_OUTPUT_FOLDER = "../../reports"
DEFAULT_LIST_OF_PARAMS: List[Dict[str, Optional[Union[str, float]]]] = [
    {"ID": None, "n": 10, "a": 1, "b": 1, "title": "Positive"},
    {"ID": None, "n": 15, "a": -1, "b": -1, "title": "Negative"},
    {"ID": None, "n": 20, "a": 0, "b": 2, "title": "Zero"}
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
        self._list_of_params: List[Dict[str, Optional[Union[str, float]]]]

    def set_up_params(self, notebook_path: str, output_folder: str,
                      list_of_params: List[Dict[str, Optional[Union[str, float]]]]) -> None:
        """
        Sets up the params for run. If there is specified in config not to do it, it returns default values,
        otherwise it gets param from config.
        :param notebook_path: str.
        :param output_folder: str.
        :param list_of_params: str.
        """
        if self._config.get_data().param_ntb_execution.use_default:
            self._ntb_path = notebook_path
            self._output_folder = output_folder
            self._list_of_params = list_of_params
        else:
            self._ntb_path = self._config.get_data().param_ntb_execution.ntb_path
            self._output_folder = self._config.get_data().param_ntb_execution.output_folder
            self._list_of_params = self._config.get_data().param_ntb_execution.notebook_executioner_params

    def execute(self, notebook_name: str = "notebook", keep_name_static: bool = False, add_datetime_id: bool = True,
                add_file_name_to_notebook_name: bool = False, add_params_to_name: bool = False,
                convert_to_html: bool = True) \
            -> None:
        """
        Executes the notebook based on default params or config params.
        :param notebook_name: str. First part of notebook name.
        :param keep_name_static: bool. If True, the name of notebooks won't change during the execution.
        :param add_datetime_id: bool. If to add datetime id to the beginning of the file name.
        :param add_file_name_to_notebook_name: bool. If True, adds file name to notebook name.
        :param add_params_to_name: bool. If True, parameters are added to the end of the notebook.
        :param convert_to_html: bool. If to convert to html or not.
        """

        # n = 0
        for params in self._list_of_params:
            if keep_name_static:
                path_out = os.path.abspath(os.path.join(self._output_folder,
                                                        f"{notebook_name}_{os.path.basename(__file__)[:-3]}.ipynb"))
            else:
                name = notebook_name
                # add_file_name_to_notebook_name can be outside the loop, but I am keeping it here for better
                # overview of file name creation
                if add_file_name_to_notebook_name:
                    name = f"{name}_{os.path.basename(__file__)[:-3]}"
                if add_datetime_id:
                    datetime_id = create_datetime_id(now=datetime.now(), add_micro=False)
                    name = f"{datetime_id}_{name}"
                    params["ID"] = datetime_id
                if add_params_to_name:
                    for key, value in params.items():
                        if key != "ID":
                            name = f"{name}_{value}"
                path_out = os.path.abspath(os.path.join(self._output_folder, f"{name}.ipynb"))
                # Not necessary, unique name can be created by adding file name and datetime id. Keeping it here for
                # case of need in the future.
                # if name_with_number:
                #     n = n + 1
                #     name = f"{notebook_name}_{add_zeros_in_front_and_convert_to_string(n, 10000)}"
            papermill.execute_notebook(self._ntb_path, path_out, params)
            if convert_to_html:
                os.system("jupyter nbconvert --to html " + path_out)


if __name__ == "__main__":
    TIMER = Timer()

    # RUN DEFINITION ---------------------------------------------------------------------------------------------------
    NOTEBOOK_NAME = "notebook"
    KEEP_NAME_STATIC = False
    ADD_DATETIME_ID = True
    ADD_FILE_NAME_TO_NOTEBOOK_NAME = True
    ADD_PARAMS_TO_NAME = True
    CONVERT_TO_HTML = True
    # RUN DEFINITION ---------------------------------------------------------------------------------------------------

    CONFIG_NAME = None  # None
    EXECUTIONER = ParamNotebookExecutioner(CONFIG_NAME)

    OUTPUT_FOLDER = Config().get_data().path.auto_notebooks
    # OUTPUT_FOLDER = DEFAULT_OUTPUT_FOLDER

    EXECUTIONER.set_up_params(
        notebook_path=DEFAULT_NTB_PATH,
        output_folder=OUTPUT_FOLDER,
        list_of_params=DEFAULT_LIST_OF_PARAMS
    )
    TIMER.start()
    EXECUTIONER.execute(
        notebook_name=NOTEBOOK_NAME,
        keep_name_static=KEEP_NAME_STATIC,
        add_datetime_id=ADD_DATETIME_ID,
        add_file_name_to_notebook_name=ADD_FILE_NAME_TO_NOTEBOOK_NAME,
        add_params_to_name=ADD_PARAMS_TO_NAME,
        convert_to_html=CONVERT_TO_HTML
    )
    TIMER.end(label="End of Notebook Executioner")
