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

# # Data Frame Explorer Documentation
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
#     - [Data Frame](#2-1)   
#     - [Exploration](#2-2)  
#         - [Info about Data Frame](#2-2-1)
#         - [Attribute Types](#2-2-2)
#         - [Memory Usage](#2-2-3)
#         - [NaN Statistics](#2-2-4)
#         - [Attribute Statistics](#2-2-5)
#     - [Data Frames Comparison](#2-3)
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

from importlib import reload
from pandas import DataFrame
from numpy.random import choice, randn, seed

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)   
# Code, libraries, classes, functions from within the repository.

import src.data.df_explorer as DFE

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

# #### Constants for Setting Automatic Run
# [ToC](#ToC)   



# #### Notebook Specific Constants
# [ToC](#ToC)   



# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)   

config = Config().get()

# <a name="2-1"></a>
# ## Data Frame
# [ToC](#ToC)   


# +
df = DataFrame()
n = 100
seed(876)
df["sex"] = choice(["male", "female"], n)
df["number"] = randn(n)

df.head()
# -

# <a name="2-2"></a>
# ## Exploration
# [ToC](#ToC)   


reload(DFE)
df_explorer = DFE.DFExplorer()

# <a name="2-2-1"></a>
# ###  Info about Data Frame
# [ToC](#ToC)   

df_explorer.print_info_about_data_frame(df=df)

# <a name="2-2-2"></a>
# ###  Attribute Types
# [ToC](#ToC)  

df_explorer.get_df_types(df=df)

# <a name="2-2-3"></a>
# ###  Memory Usage
# [ToC](#ToC)  

df_explorer.get_memory_usage(df=df, attr_name="number", list_dtypes=["float64", "float32", "float16"])

# <a name="2-2-4"></a>
# ###  NaN Statistics
# [ToC](#ToC)  

df_explorer.get_nan_stats(df=df, fraction=True)

df_explorer.get_nan_stats(df=df, fraction=False)

# <a name="2-2-5"></a>
# ###  Attribute Statistics
# [ToC](#ToC)  

df_explorer.print_attr_stats(df=df)

# <a name="2-3"></a>
# ## Data Frames Comparison
# [ToC](#ToC)   

data_1 = [
    [1., 2., 3.],
    [3., 2., 1.],
    [4., 5., 2.]
]
data_2 = [
    [1., 2., 3.],
    [3., 10., 1.],
    [4., 5., 2.]
]
attr_names = ["NUMBER_1", "NUMBER_2", "NUMBER_3"]
df_1 = DataFrame(data_1, columns=attr_names)
df_2 = DataFrame(data_2, columns=attr_names)

# identical data frames
df_explorer.compare_attributes_in_data_frames(df_1, df_1, attr_names)

# not identical data frames
df_explorer.compare_attributes_in_data_frames(df_1, df_2, attr_names)

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)   

Logger().end_timer()
