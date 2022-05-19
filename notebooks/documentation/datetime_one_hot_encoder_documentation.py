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

# # Datetime One Hot Encoder Documentation
# *Version:* `1.0` *(Jupytext, time measurements, logger)*

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
#     - [Input Data](#2-1)   
#     - [One Hot Transformation](#2-2) 
#         - [Basic Fit and Prediction](#2-2-1)
#         - [Full Prediction Without Minutes](#2-2-2)
#         - [Minutes Prediction](#2-2-3)
# - [Final Timestamp](#3)  

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC) 

# Notebook for documentation of one hot encoder. 

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

from src.utils.notebook_support_functions import create_button, get_notebook_name
from src.utils.logger import Logger
from src.utils.envs import Envs
from src.utils.config import Config
from pandas import options
from IPython.display import display, HTML

# > Constants for overall behaviour.

LOGGER_CONFIG_NAME = "logger_file_console" # default
PYTHON_CONFIG_NAME = "python_local" # default
CREATE_BUTTON = False
ADDAPT_WIDTH = False
NOTEBOOK_NAME = get_notebook_name()

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
# [ToC](#ToC)  

from datetime import datetime
from pandas import DatetimeIndex
from pprint import pprint

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

from src.transformations.datetime_one_hot_transformer import DatetimeOneHotEncoderTransformer
from tests.tests_transformations.test_datetime_one_hot_transformer import INPUT_DATA

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
CRY_CONFIG_NAME = "cry_basic"

# #### Constants for Setting Automatic Run
# [ToC](#ToC)  



# #### Notebook Specific Constants
# [ToC](#ToC)  



# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

# <a name="2-1"></a>
# ## Unput Data
# [ToC](#ToC)  


pprint(INPUT_DATA)

# <a name="2-2"></a>
# ## One Hot Transformation
# [ToC](#ToC)  


transformer = DatetimeOneHotEncoderTransformer()

# <a name="2-2-1"></a>
# ### Basic Fit and Prediction
# [ToC](#ToC)  

# +
output = transformer.fit_predict(
    dt_index=INPUT_DATA, 
    add_hours=False, 
    add_days_of_week=True, 
    add_weekend=True, 
    add_months=True, 
    add_years=False, 
    min_interval=0
)
output_names = transformer.get_encoded_attribute_names()

pprint(output)
pprint(output_names)
# -

print(f"Prediction with November not in training data:")
print(output_names)
print(transformer.predict(DatetimeIndex([datetime(2016, 11, 1, 0, 1, 0)])))

print(f"Prediction with December in training data:")
print(output_names)
print(transformer.predict(DatetimeIndex([datetime(2016, 12, 1, 0, 1, 0)])))

# <a name="2-2"></a>
# ### Full Prediction Without Minutes
# [ToC](#ToC)  

# +
output = transformer.fit_predict(
    dt_index=INPUT_DATA, 
    add_hours=True, 
    add_days_of_week=True, 
    add_weekend=True, 
    add_months=True, 
    add_years=True, 
    min_interval=0
)
output_names = transformer.get_encoded_attribute_names()

pprint(output)
pprint(output_names)
# -

# <a name="2-2-3"></a>
# ### Minutes Prediction
# [ToC](#ToC)  

# +
output = transformer.fit_predict(
    dt_index=INPUT_DATA, 
    add_hours=False, 
    add_days_of_week=False, 
    add_weekend=False, 
    add_months=False, 
    add_years=False, 
    min_interval=120
)
output_names = transformer.get_encoded_attribute_names()

pprint(output)
pprint(output_names)
# -

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
