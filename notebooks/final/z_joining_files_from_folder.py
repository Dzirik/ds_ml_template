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

# # Joining Files from Folder
# *Version:* `1.2` *(Jupytext, time measurements, logger, param notebook execution, fixes)*

# This notebook helps to join data frames from many files in a folder.

# <a name="0"></a>
# ## Table of Content
#
# - [General Settings](#1)
#     - [Paths](#1-1)
#     - [Notebook Functionality and Appearance](#1-2)
#     - [External Libraries](#1-3)
#     - [Internal Code](#1-4)
#     - [Constants](#1-5)   
# - [Analysis](#2)  
#     - [Setting Up the Run](#2-1)  
#     - [Joining Data Frames](#2-2)
#     - [Saving Data Frame](#2-inf)
# - [Final Timestamp](#3)  

# <a name="1"></a>
# # GENERAL SETTINGS
# [ToC](#0)  
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
import os
from pandas import DataFrame, read_csv, concat
from numpy import array, matrix

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#0)  
# Code, libraries, classes, functions from within the repository.

from src.utils.date_time_functions import create_datetime_id
from src.data.saver_and_loader import SaverAndLoader
from src.utils.date_time_functions import convert_datetime_to_string_date

# <a name="1-5"></a>
# ### Constants
# [ToC](#0)  
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



# <a name="2"></a>
# # ANALYSIS
# [ToC](#0)  

config_data = Config().get_data()

# <a name="2-1"></a>
# ## Setting Up the Run
# [ToC](#0)  

FOLDER_PATH = "E:/TEMP/JOIN"

FILE_NAME_FOR_SAVE = ""

# <a name="2-2"></a>
# ## Joining Data Frames
# [ToC](#0)  


# +
directory = os.fsencode(FOLDER_PATH)
  
df = None
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"): # and filename.endswith("evaluation.csv"): 
        # print(os.path.join(FOLDER_PATH, filename))
        file_path = os.path.join(FOLDER_PATH, filename)
        if df is None:
            df = read_csv(file_path, sep=",", decimal=".")
        else:
            df = concat([df, read_csv(file_path, sep=",", decimal=".")])
    else:
        continue

if df is not None:
    df.reset_index(drop=True, inplace=True)
    print(df.shape)
    # df = df.drop_duplicates()
    print(df.shape)
else:
    df = DataFrame()
# -

df.head(10)

# <a name="2-inf"></a>
# ## Saving Data Frame
# [ToC](#0)  

saver_and_loader = SaverAndLoader()

if df.shape[0] > 0:
    file_name = f"{FILE_NAME_FOR_SAVE}_{convert_datetime_to_string_date()}_JOINED_RESULTS"
    saver_and_loader.save_dataframe_to_csv(df, file_name, where="results_data")

# <a name="3"></a>
# # Final Timestamp
# [ToC](#0)  

Logger().end_timer()
