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

# # Anomaly Detection Sample Data Documentation
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
#     - [Generate One Data Frame](#2-1)
#         - [Data Generation in Default Configuration](#2-1-1)    
#         - [Other Configuration](#2-1-2)
#     - [Generate List of Data Frames](#2-2)
# - [Final Timestamp](#3)  

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC) 

# Sample data sets for testing anomaly detection pipelines and algorithms. 
#
# General settings:
# - `abs_values` - If values are bigger or equal to zero.
# - `add_zeros` - If to add zeros. This is especially useful together with `abs_value` for gesting logaritmic.
#
# In general, the data set has to have following structure:
# - Datetime index as data frame index.
# - Attribute `ATTR_DATE_TIME`. Datetime index as the index column.
# - Attribute `ATTR_ID` as an integer identifier of the observation. This is necessary for cross validation and models and pipelines assessment for better work than with data time format.

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

from src.utils.date_time_functions import create_datetime_id
from src.data.anomaly_detection_sample_data import AnomalyDetectionSampleData, plot_data_frame_series
from src.data.attributes import A
from src.data.anomaly_detection_sample_data import ATTRS

# <a name="1-5"></a>
# ### Constants
# [ToC](#ToC)  
# Constants for the notebook.
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

ABS_VALUES = True
ADD_ZEROS = False
# -

# #### Python Config Initialisation
# [ToC](#ToC)

envs.set_config(PYTHON_CONFIG_NAME)

# #### Notebook Specific Constants
# [ToC](#ToC)



# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

# <a name="2-1"></a>
# ## Generate One Data Frame
# [ToC](#ToC)  

# <a name="2-1-1"></a>
# ### Data Generation in Default Configuration
# [ToC](#ToC)  


data = AnomalyDetectionSampleData(abs_values=ABS_VALUES, add_zeros=ADD_ZEROS)

df = data.generate(n=100, seed_number=376)

df.head()

plot_data_frame_series(df, ATTRS)

# <a name="2-1-2"></a>
# ### Other Configuration
# [ToC](#ToC)  

# +
data = AnomalyDetectionSampleData(abs_values=True, add_zeros=True)
df = data.generate(n=100, seed_number=376)

print(df.head(N_ROWS_TO_DISPLAY))

plot_data_frame_series(df, ATTRS)

# +
data = AnomalyDetectionSampleData(abs_values=False, add_zeros=True)
df = data.generate(n=100, seed_number=376)

print(df.head(N_ROWS_TO_DISPLAY))

plot_data_frame_series(df, ATTRS)

# +
data = AnomalyDetectionSampleData(abs_values=True, add_zeros=False)
df = data.generate(n=100, seed_number=376)

print(df.head(N_ROWS_TO_DISPLAY))

plot_data_frame_series(df, ATTRS)

# +
data = AnomalyDetectionSampleData(abs_values=False, add_zeros=False)
df = data.generate(n=100, seed_number=376)

print(df.head(N_ROWS_TO_DISPLAY))

plot_data_frame_series(df, ATTRS)
# -

# <a name="2-2"></a>
# ## Generate List of Data Frames
# [ToC](#ToC) 

data = AnomalyDetectionSampleData(abs_values=True, add_zeros=True)
dfs, df_whole = data.create_list_of_data_frames(n=10, seed_number=398)

dfs

df_whole

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
