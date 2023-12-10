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

# # Anomaly Detection Rule
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
#     - [WECO Rules](#2-1)   
#         - [First WECO Rule](#2-1-1)     
#         - [Second WECO Rule](#2-1-2) 
#         - [Thirt WECO Rule](#2-1-3) 
#         - [Fourth WECO Rule](#2-1-4) 
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
from numpy import array, apply_along_axis

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

from src.utils.date_time_functions import create_datetime_id
from src.models.anomaly_detection_rules import FirstWecoRule, SecondWecoRule, ThirdWecoRule, FourthWecoRule

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

std_est = 1.

# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

# <a name="2-1"></a>
# ## WECO Rules
# [ToC](#ToC)  


# <a name="2-1-1"></a>
# ### First WECO Rule
# [ToC](#ToC)  


residuals = array([
    [9.],
    [4.],
    [0.],
    [-2.],
    [-9.]
])

rule = FirstWecoRule()

index = 0
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 2
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 2
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 4
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

apply_along_axis(func1d=rule.apply, axis=1, arr=residuals, std_est=std_est, side="both", alpha=1.)

# <a name="2-1-2"></a>
# ### Second WECO Rule
# [ToC](#ToC)  

residuals = array([
    [3., 1., 9.],
    [4., 0., 4.],
    [-4., -1., -8.],
    [0., 0., 0.],
    [-8., 0., -3.]
])

rule = SecondWecoRule()

index = 0
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 1
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 2
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 3
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

apply_along_axis(func1d=rule.apply, axis=1, arr=residuals, std_est=std_est, side="both", alpha=1.)

# <a name="2-1-3"></a>
# ### Third WECO Rule
# [ToC](#ToC)  

residuals = array([
    5 * [2.],
    5 * [-2],
    5 * [0.]
])

rule = ThirdWecoRule()

index = 0
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 1
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 2
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

apply_along_axis(func1d=rule.apply, axis=1, arr=residuals, std_est=std_est, side="both", alpha=1.)

# <a name="2-1-4"></a>
# ### Fourth WECO Rule
# [ToC](#ToC)  

residuals = array([
    8 * [0.2],
    8 * [-0.2],
    8 * [0.]
])

rule = FourthWecoRule()

index = 0
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 1
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

index = 2
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="both", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="positive", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")
is_rule = rule.apply(residuals=residuals[index,], std_est=std_est, side="negative", alpha=1.)
print(f"For Residuals: {residuals[index,]} the output is: {is_rule}")

apply_along_axis(func1d=rule.apply, axis=1, arr=residuals, std_est=std_est, side="both", alpha=1.)

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
