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

# # Low Level Tools Documentation
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
#     - [Environmental Variables](#2-1)   
#     - [Environment Configuration](#2-2)     
#     - [Timer](#2-3)
#     - [Logger](#2-4)
# - [Final Timestamp](#3)  

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC) 

# This notebooks serves for documentation of low-level repository tools:
# - Environment variables handling.
# - Environment configuration.
# - Timer.
# - Logger.
#
# For general description, please see the *README.md* file.

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

# +
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from importlib import reload

# # %matplotlib notebook
# # %matplotlib inline
# -

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

# +
# Space for importing my own code.
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
CRY_CONFIG_NAME = "cry_basic"

# #### Constants for Setting Automatic Run
# [ToC](#ToC)  



# #### Notebook Specific Constants
# [ToC](#ToC)  



# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

# <a name="2-1"></a>
# ## Environmental Variables
# [ToC](#ToC)  


from src.utils.envs import Envs

envs = Envs()

# Getting the variables:

print(f"Current Environment Configuration File is: {envs.get_config()}")
print(f"Current Logger Configuration File is: {envs.get_logger()}")
print(f"Current Unit Testing Setting is: {envs.get_running_unit_tests()}")

# Setting and getting the variables:

envs.set_config("python_repo")
envs.get_config()

# <a name="2-2"></a>
# ## Environment Configuration
# [ToC](#ToC) 


from src.utils.config import Config

# For setting the configuration file to be used, please see previous chapter.   
# Getting current config:

config_data = Config().get()

config_data

# Examples of access to the variables:

config_data.name

config_data.path.raw_data

# <a name="2-3"></a>
# ## Timer
# [ToC](#ToC) 

from src.utils.timer import Timer
from time import sleep

timer = Timer()

# Set if print the results:

timer.set_results_printing(print_results=True)

# Starting the timer:

timer.start()

# Adding several meantimes:

sleep(1)
timer.set_meantime(label="First meantime.")
sleep(2)
timer.set_meantime(label="Second meantime.")
sleep(3)
timer.set_meantime(label="Third meantime.")
sleep(4)

# End the timer:

timer.end(label="Last interval.")

(MT, MT_C, MT_L, df) = timer.get_data()

print(MT)

print(MT_C)

print(MT_L)

df

# <a name="2-4"></a>
# ## Logger
# [ToC](#ToC) 

from src.utils.logger import Logger

# Setting only console logger to see the output here in Jupyter notebook.

NON_DEFAULT_LOGGER_CONFIG_NAME = "logger_console"
envs.set_logger(NON_DEFAULT_LOGGER_CONFIG_NAME)

# Taking the logger:

logger = Logger()

# Log all the events:

logger.debug("debug")
logger.info("info")
logger.warning("warning")
logger.error("error")
logger.critical("error")

# Time measurements:

logger.start_timer("Simple timer test")
sleep(0.1)
logger.set_meantime("First interval")
sleep(0.2)
logger.set_meantime("Second interval")
logger.end_timer()

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
