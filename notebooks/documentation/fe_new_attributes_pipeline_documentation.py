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

# # New Attributes Pipeline Documentation
# *Version:* `1.2` *(Jupytext, time measurements, logger, param notebook execution, fixes)*

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
#     - [Explicit Configuration Definition](#2-2)
#     - [Using Configuration From Configuration File](#2-3)
#     - [Results Comparison](#2-4)
# - [Final Timestamp](#3)  

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC) 

# In this notebook, only artifially generated data is used to show the usage of the code. Please see other notebooks for usage on production data.

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

# #### Overall Setting Specification

LOGGER_CONFIG_NAME = "logger_file_limit_console"
ADDAPT_WIDTH = False

# #### Overall Behaviour Setting

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

options.display.max_rows = 500
options.display.max_columns = 500
envs = Envs()
envs.set_logger(LOGGER_CONFIG_NAME)
Logger().start_timer(f"NOTEBOOK; Notebook name: {NOTEBOOK_NAME}")
if ADDAPT_WIDTH:
    display(HTML("<style>.container { width:100% !important; }</style>")) # notebook width

# +
# create_button()
# -

# <a name="1-3"></a>
# ### External Libraries
# [ToC](#ToC)  

from datetime import datetime

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

# +
from src.utils.date_time_functions import create_datetime_id

from src.data.time_series_one_minute_data import TimeSeriesOneMinuteData

from src.pipelines.new_attributes_pipeline import NewAttributesPipeline

from src.pipelines.new_attributes_pipeline_config_data import NewAttributesPipelineConfigData, NewAttribute
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

# + tags=["parameters"]
# MANDATORY FOR CONFIG DEFINITION AND NOTEBOOK AND ITS OUTPUTS IDENTIFICATION #########################################
PYTHON_CONFIG_NAME = "python_local"
ID = create_datetime_id(now=datetime.now(), add_micro=False)
# (END) MANDATORY FOR CONFIG DEFINITION AND NOTEBOOK AND ITS OUTPUTS IDENTIFICATION ###################################
# -

# #### Python Config Initialisation
# [ToC](#ToC)

envs.set_config(PYTHON_CONFIG_NAME)

# #### Notebook Specific Constants
# [ToC](#ToC)  

CONFIG_FILE_NAME = "pipeline_new_attributes_documentation"

# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

# <a name="2-1"></a>
# ## Data Reading
# [ToC](#ToC)  


ts_data = TimeSeriesOneMinuteData()
df = ts_data.get_data_frame()
attrs = ts_data.get_attrs()

df.head(N_ROWS_TO_DISPLAY)

df.tail(N_ROWS_TO_DISPLAY)

# <a name="2-2"></a>
# ## Explicit Configuration Definition
# [ToC](#ToC)  
#
# The definition is as follows:
# ~~~
# explicit_config = NewAttributesPipelineConfigData(
#     name="explicit_config",
#     new_attributes=[
#         NewAttribute(
#             create=True,
#             name_of_transformation="datetime_family",
#             for_attrs=[],
#             use_saved=False,
#             name_of_file="",
#             parameters={"add_hours": True, "add_days_of_week": True}
#         )
#     ]
# )
# ~~~


explicit_config = NewAttributesPipelineConfigData(
    name="explicit_config",
    new_attributes=[
        NewAttribute(
            create=True,
            name_of_transformation="diff",
            for_attrs=["2*I"],
            use_saved=False,
            name_of_file="",
            parameters={}
        ),
        NewAttribute(
            create=True,
            name_of_transformation="diff_perc",
            for_attrs=["1"],
            use_saved=False,
            name_of_file="",
            parameters={}
        ),
        NewAttribute(
            create=True,
            name_of_transformation="datetime_family",
            for_attrs=[],
            use_saved=False,
            name_of_file="",
            parameters={"add_hours": True, "add_days_of_week": True}
        )
    ]
)

# +
config_file_name = None # CONFIG_FILE_NAME

pipeline = NewAttributesPipeline(config_file_name)
pipeline.set_config_data(explicit_config)

df_out_explicit = pipeline.execute(df.copy())
# -

df_out_explicit.head(N_ROWS_TO_DISPLAY)

df_out_explicit.tail(N_ROWS_TO_DISPLAY)

# <a name="2-3"></a>
# ## Using Configuration From Configuration File
# [ToC](#ToC) 

# +
config_file_name = CONFIG_FILE_NAME

pipeline = NewAttributesPipeline(config_file_name)

df_out_from_file = pipeline.execute(df.copy())
# -

df_out_from_file.head(N_ROWS_TO_DISPLAY)

df_out_from_file.tail(N_ROWS_TO_DISPLAY)

# <a name="2-4"></a>
# ## Results Comparison
# [ToC](#ToC) 

assert df_out_explicit.equals(df_out_from_file)

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
