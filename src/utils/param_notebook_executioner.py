"""
Code for automatic parameterized notebook execution.
- Based on papermill library.
- The notebook has to be in *.ipynb instead of *.py.
- The script works being run from PyCharm. There is a problem with paths running it from console.
Version: 1.2
- Parallelization.
"""
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from random import shuffle
from typing import List, Dict, Union, Optional, Any

import papermill

from src.utils.config import Config
from src.utils.date_time_functions import create_datetime_id
from src.utils.envs import Envs
from src.utils.timer import Timer

# THESE PARAMETERS ARE IN CONFIG FILE AS WELL, AS DEFAULT ONES THE ONES HERE ARE USED ##################################
DEFAULT_NTB_PATH = "../../notebooks/template/template_parameterized_execution_notebook.ipynb"
DEFAULT_OUTPUT_FOLDER = None  # "../../reports", if None, auto_notebooks folder from config path will be used
# (END) THESE PARAMETERS ARE IN CONFIG FILE AS WELL, AS DEFAULT ONES THE ONES HERE ARE USED ############################

# NOTEBOOKS RUN SETTINGS ###############################################################################################
NOTEBOOK_NAME = "notebook"
KEEP_NAME_STATIC = False
ADD_DATETIME_ID = True
ADD_FILE_NAME_TO_NOTEBOOK_NAME = True
ADD_PARAMS_TO_NAME = False
CONVERT_TO_HTML = True
# (END) NOTEBOOKS RUN SETTINGS #########################################################################################

# PARALLELIZATION SETTING ##############################################################################################
NUMBER_OF_PROCESSES = 3  # None, 3
SHUFFLE_BEFORE_RUNNING = True
# (END) PARALLELIZATION SETTING ########################################################################################

# PARAMETERS SETTINGS ##################################################################################################
PYTHON_CONFIG_NAME = "python_personal"  # None

N_SLEEP_SECONDSS = [
    (10, 11),
    (15, 19),
    (20, 29)
]

A_B_TITLE_MODEL_CLASS_TYPE_MODEL_PARAMS_LIST = [
    (1, 1, "Positive", "LinearModel", [["intercept", True]]),
    (-1, -1, "Negative", "LassoModel", [["alpha", 0.5], ["max_iter", 500]]),
    (0, 1, "Zero", "LassoModel", [["alpha", 0.75], ["max_iter", 1000]])
]

DEFAULT_LIST_OF_PARAMS: List[Dict[str, Optional[Union[str, int, float, List[Any], List[List[Any]]]]]] = []
for n, sleep_seconds in N_SLEEP_SECONDSS:
    for a, b, title, model_class_type, model_params_list in A_B_TITLE_MODEL_CLASS_TYPE_MODEL_PARAMS_LIST:
        DEFAULT_LIST_OF_PARAMS.append({
            "PYTHON_CONFIG_NAME": PYTHON_CONFIG_NAME,
            "ID": None,
            "n": n,
            "a": a,
            "b": b,
            "title": title,
            "sleep_seconds": sleep_seconds,
            "MODEL_CLASS_TYPE": model_class_type,
            "MODEL_PARAMS_LIST": model_params_list
        })


# when run with no threading, the execution time is: 4.98 m
# when run with 3 threads, the execution time is: 2.18 m in no shuffle mode
# (END) PARAMETERS SETTINGS ############################################################################################


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
        self._list_of_params: List[Dict[str, Optional[Union[str, int, float, List[Any], List[List[Any]]]]]]

    def set_up_params(self, notebook_path: str, output_folder: str,
                      list_of_params: List[Dict[str, Optional[Union[str, int, float, List[Any], List[List[Any]]]]]]) \
            -> None:
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
            datetime_id = create_datetime_id(now=datetime.now(), add_micro=False)
            params["ID"] = datetime_id
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
                    name = f"{datetime_id}_{name}"
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


def execute_for_params(list_of_params: List[Dict[str, Optional[Union[str, int, float, List[Any], List[List[Any]]]]]]) \
        -> None:
    """
    Executes for list of parameters. This function is here because of multiprocessing.
    :param list_of_params: List[Dict[str, Optional[Union[str, int, float, List[Any], List[List[Any]]]]]].
    """
    executioner = ParamNotebookExecutioner(PYTHON_CONFIG_NAME)

    if DEFAULT_OUTPUT_FOLDER is None:
        output_folder = Config().get_data().path.auto_notebooks
    else:
        output_folder = DEFAULT_OUTPUT_FOLDER

    executioner.set_up_params(
        notebook_path=DEFAULT_NTB_PATH,
        output_folder=output_folder,
        list_of_params=list_of_params
    )
    executioner.execute(
        notebook_name=NOTEBOOK_NAME,
        keep_name_static=KEEP_NAME_STATIC,
        add_datetime_id=ADD_DATETIME_ID,
        add_file_name_to_notebook_name=ADD_FILE_NAME_TO_NOTEBOOK_NAME,
        add_params_to_name=ADD_PARAMS_TO_NAME,
        convert_to_html=CONVERT_TO_HTML
    )


if __name__ == "__main__":
    TIMER = Timer()

    TIMER.start()
    print(f"\n{'#' * 50} EXECUTING FOR {NUMBER_OF_PROCESSES} PROCESSES {'#' * 50}")
    print(f" - number of threads is: {cpu_count()}")
    print(f" - running {NUMBER_OF_PROCESSES} processes")
    if NUMBER_OF_PROCESSES is None:
        execute_for_params(DEFAULT_LIST_OF_PARAMS)
    else:
        if SHUFFLE_BEFORE_RUNNING:
            shuffle(DEFAULT_LIST_OF_PARAMS)
        if len(DEFAULT_LIST_OF_PARAMS) < NUMBER_OF_PROCESSES:
            NUMBER_OF_PROCESSES = len(DEFAULT_LIST_OF_PARAMS)
            print(f" - updating number of processes to: {NUMBER_OF_PROCESSES} "
                  f"for length: {len(DEFAULT_LIST_OF_PARAMS)}")
        BATCH_SIZE = len(DEFAULT_LIST_OF_PARAMS) // NUMBER_OF_PROCESSES
        params_batches = [
            DEFAULT_LIST_OF_PARAMS[i:i + BATCH_SIZE] for i in range(0, len(DEFAULT_LIST_OF_PARAMS), BATCH_SIZE)
        ]

        with Pool(NUMBER_OF_PROCESSES) as notebook_pool:
            notebook_pool.map(execute_for_params, params_batches)
            notebook_pool.close()
            notebook_pool.join()

    TIMER.end(label="End of Notebook Executioner")
