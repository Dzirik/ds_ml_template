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
# *Version:* `1.0` *(Jupytext, time measurements, logger)*

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
# [ToC](#0)  
#
# Adding paths that are necessary to import code from within the repository.

import sys
import os
sys.path+=[os.path.join(os.getcwd(), ".."), os.path.join(os.getcwd(), "../..")] # one and two up

# <a name="1-2"></a>
# ### Notebook Functionality and Appearance
# [ToC](#0)  
# Necessary libraries for notebook functionality:
# - A button for hiding/showing the code. By default it is deactivated and can be activated by setting CREATE_BUTTON constant to True. 
# > **NOTE: This way, using the function, the button works only in active notebook. If the functionality needs to be preserved in html export, then the code has to be incluced directly into notebook.**
# - Set notebook width to 100%.
# - Notebook data frame setting for better visibility.
# - Initial timestamp setting and logging the start of the execution.

from src.utils.notebook_support_functions import create_button, get_notebook_name
from src.utils.logger import Logger
from src.utils.envs import Envs
from src.utils.config import Config
from pandas import options
from IPython.core.display import display, HTML

# > Constants for overall behaviour.

LOGGER_CONFIG_NAME = "logger_file_console" # default
PYTHON_CONFIG_NAME = "python_personal" # default
CREATE_BUTTON = False
ADDAPT_WIDTH = False
# doesnt work with auto notebook run
try:
    from src.utils.notebook_support_functions import create_button, get_notebook_name
    NOTEBOOK_NAME = get_notebook_name()
except:
    NOTEBOOK_NAME = "joining_files_from_folder"

options.display.max_rows = 500
options.display.max_columns = 500
envs = Envs()
envs.set_logger(LOGGER_CONFIG_NAME)
envs.set_config(PYTHON_CONFIG_NAME)
Logger().start_timer(f"NOTEBOOK; Notebook name: {NOTEBOOK_NAME}")
if CREATE_BUTTON:
    create_button()
if ADDAPT_WIDTH:
    display(HTML("<style>.container { width:100% !important; }</style>")) # notebook width

# <a name="1-3"></a>
# ### External Libraries
# [ToC](#0)  

import os
from pandas import DataFrame, read_csv, concat
from numpy import array, matrix

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#0)  
# Code, libraries, classes, functions from within the repository.

from src.data.saver_and_loader import SaverAndLoader
from src.utils.date_time_functions import convert_datetime_to_string_date

# <a name="1-5"></a>
# ### Constants
# [ToC](#0)  
# Constants for the notebook.
#
# > *NOTE: Please use all letters upper.*

# #### General Constants
# [ToC](#0)  

# from src.global_constants import *  # Remember to import only the constants in use
N_ROWS_TO_DISPLAY = 2
FIGURE_SIZE_SETTING = {"autosize": False, "width": 2200, "height": 750}
DATA_PROCESSING_CONFIG_NAME = "data_processing_basic"

# #### Constants for Setting Automatic Run
# [ToC](#0)  



# #### Notebook Specific Constants
# [ToC](#0)  



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

df.reset_index(drop=True, inplace=True)
print(df.shape)
# df = df.drop_duplicates()
print(df.shape)
# -

df.head(10)

# <a name="2-inf"></a>
# ## Saving Data Frame
# [ToC](#0)  

saver_and_loader = SaverAndLoader()

file_name = f"{FILE_NAME_FOR_SAVE}_{convert_datetime_to_string_date()}_JOINED_RESULTS"
saver_and_loader.save_dataframe_to_csv(df, file_name, where="results_data")

# <a name="3"></a>
# # Final Timestamp
# [ToC](#0)  

Logger().end_timer()
