# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Row Blocks Grouping Pipeline Documentation
# *Version:* `1.1` *(Jupytext, time measurements, logger, param notebook execution)*

# <a name="ToC"></a>
# # Table of Content
#
# - [Notebook Description](#0)
# - [General Settings](#1)
#     - [Paths](#1-1)
#     - [Notebook Functionality and Appearance](#1-2)
#     - [External Libraries](#1-3)
#     - [Internal Code](#1-4)
#     - [Constants](#1-5)   
# - [Analysis](#2)   
#     - [Data Reading](#2-1)     
#     - [Pipeline Execution from By Hand Config](#2-2)
#     - [Pipeline Execution from File Configuration](#2-3)
#     - [Results Comparison](#2-4)
# - [Final Timestamp](#3)  

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC) 

# > *Please put your comments about the notebook functionality here.*  

# <a name="1"></a>
# # GENERAL SETTINGS
# [ToC](#ToC)  
# General settings for the notebook (paths, python libraries, own code, notebook constants). 
#
# > *NOTE: All imports and constants for the notebook settings shoud be here. Nothing should be imported in the analysis section.*

# <a name="1-1"></a>
# ### Paths
# [ToC](#ToC)  
#
# Adding paths that are necessary to import code from within the repository.

import sys
import os
sys.path+=[os.path.join(os.getcwd(), ".."), os.path.join(os.getcwd(), "../..")] # one and two up

# <a name="1-2"></a>
# ### Notebook Functionality and Appearance
# [ToC](#ToC)  
# Necessary libraries for notebook functionality:
# - A button for hiding/showing the code. By default it is deactivated and can be activated by setting CREATE_BUTTON constant to True. 
# > **NOTE: This way, using the function, the button works only in active notebook. If the functionality needs to be preserved in html export, then the code has to be incluced directly into notebook.**
# - Set notebook width to 100%.
# - Notebook data frame setting for better visibility.
# - Initial timestamp setting and logging the start of the execution.

try:
    from src.utils.notebook_support_functions import create_button, get_notebook_name
    NOTEBOOK_NAME = get_notebook_name()
    SUPPORT_FUNCTIONS_READ = True
except:
    NOTEBOOK_NAME = "NO_NAME"
    SUPPORT_FUNCTIONS_READ = False  

from src.utils.logger import Logger
from src.utils.envs import Envs
from src.utils.config import Config
from pandas import options
from IPython.display import display, HTML

# > Constants for overall behaviour.

LOGGER_CONFIG_NAME = "logger_file_console" # default
PYTHON_CONFIG_NAME = "python_personal" # default
CREATE_BUTTON = False
ADDAPT_WIDTH = False

options.display.max_rows = 500
options.display.max_columns = 500
envs = Envs()
envs.set_logger(LOGGER_CONFIG_NAME)
envs.set_config(PYTHON_CONFIG_NAME)
Logger().start_timer(f"NOTEBOOK; Notebook name: {NOTEBOOK_NAME}")
if SUPPORT_FUNCTIONS_READ and CREATE_BUTTON:
    create_button()
if SUPPORT_FUNCTIONS_READ and ADDAPT_WIDTH:
    display(HTML("<style>.container { width:100% !important; }</style>")) # notebook width

# <a name="1-3"></a>
# ### External Libraries
# [ToC](#ToC)  



# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

# +
from src.data.time_series_one_minute_data import TimeSeriesOneMinuteData

from src.data.df_explorer import DFExplorer

from src.pipelines.row_blocks_grouping_pipeline import RowBlocksGroupingPipeline
from src.pipelines.row_blocks_grouping_pipeline_config_data import RowBlocksGroupingPipelineConfigData, RowBlocksGrouping
# -

# <a name="1-5"></a>
# ### Constants
# [ToC](#ToC)  
# Constants for the notebook.
#
# > *NOTE: Please use all letters upper.*

# #### General Constants
# [ToC](#ToC)  

# from src.global_constants import *  # Remember to import only the constants in use
N_ROWS_TO_DISPLAY = 2
FIGURE_SIZE_SETTING = {"autosize": False, "width": 2200, "height": 750}
DATA_PROCESSING_CONFIG_NAME = "data_processing_basic"

# #### Constants for Setting Automatic Run
# [ToC](#ToC)  



# #### Notebook Specific Constants
# [ToC](#ToC)  

CONFIG_FILE_NAME = "pipeline_row_blocks_grouping_documentation"

# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

df_explorer = DFExplorer()

# <a name="2-1"></a>
# ## Data Reading
# [ToC](#ToC)  


ts_data = TimeSeriesOneMinuteData()
df = ts_data.get_data_frame()

df.head()

df.tail()

# <a name="2-2"></a>
# ## Pipeline Execution from By Hand Config
# [ToC](#ToC)  
#
# Example of grouping:
#
# ~~~
# grouping = [
#     BlocksGrouping(
#         create=True,
#         attrs=[A.date_time.name, A.open.name],
#         fun="first",
#         rename={"FIRST": A.open.name}
#     )
# ]
# ~~~

config_data = RowBlocksGroupingPipelineConfigData(
    name="my",
    grouping_window_len="30min",
    row_blocks_grouping=[
        RowBlocksGrouping(
            create=True,
            attrs=["DATETIME", "1"],
            fun="sum",
            rename={"SUM": "SUM_1"}
        ),
        RowBlocksGrouping(
            create=True,
            attrs=["DATETIME", "1", "2"],
            fun="sum",
            rename={"SUM": "SUM_1_2"}
        ),
        RowBlocksGrouping(
            create=True,
            attrs=["DATETIME", "I", "-I"],
            fun="sum",
            rename={"SUM": "SUM_I-I"}
        ),
        RowBlocksGrouping(
            create=True,
            attrs=["DATETIME", "5*I", "10*I"],
            fun="mean",
            rename={"mean": "MEAN_5*I_10*I"}        
        )
    ]
)
config_data

config_file_name = None
pipeline = RowBlocksGroupingPipeline(config_file_name)
pipeline.set_config_data(config_data)
df_out = pipeline.execute(df)

df_out.head()

df_out.tail()

# <a name="2-3"></a>
# ## Pipeline Execution from File Configuration
# [ToC](#ToC)  

config_file_name = CONFIG_FILE_NAME
pipeline = RowBlocksGroupingPipeline(config_file_name)
pipeline.set_config_data(config_data)
df_out_from_file = pipeline.execute(df)

df_out_from_file.head()

df_out_from_file.tail()

# <a name="2-4"></a>
# ## Results Comparison
# [ToC](#ToC)  

assert df_out.equals(df_out_from_file)

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
