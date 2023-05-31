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

# # Transformations Executioner Transformer Execution
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
#     - [Data Preparation](#2-1) 
#     - [Configuration](#2-2)
#     - [Transformations Examples](#2-3)
# - [Final Timestamp](#3)  

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC) 

# Transformer | Invertible | For Only One Column
# -|-|-
# DifferenceTransformer | False | True
# DifferencePercentageChangeTransformer | False | True
# LogTransformer | True | False
# LogNegativeTransformer | True | False
# MinMaxScalingTransformer | True | False
# QuantileTransformer | False | True
# ShiftByValueTransformer | True | False

# <a name="1"></a>
# # GENERAL SETTINGS
# [ToC](#ToC)  
# General settings for the notebook (paths, python libraries, own code, notebook constants). 

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
PYTHON_CONFIG_NAME = "python_personal" # default
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

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

from src.data.anomaly_detection_sample_data import AnomalyDetectionSampleData, ATTRS, ATTR_DATE_TIME, ATTR_ID, plot_data_frame_series
from src.transformations.transformations_executioner_transformer import TransformationsExecutionerTransformer, TransformerConfiguration

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



# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

# <a name="2-1"></a>
# ## Data Preparation
# [ToC](#ToC)  

# +
df_data = AnomalyDetectionSampleData(abs_values=False, add_zeros=False).generate(n=5, seed_number=37896)

df_data[ATTRS[0]] = [float(2**i) for i in range(1, 6)]
df_data[ATTRS[1]] = [float(i) for i in range(1, 6)]
df_data[ATTRS[2]] = [float(i*2) for i in range(1, 6)]
df_data[ATTRS[3]] = [float(i*20) for i in range(1, 6)]
df_data[ATTRS[4]] = [-100., -50., 0., 50., 100.]

print(ATTR_DATE_TIME)
print(ATTR_ID)
print(ATTRS)
# -

df_data

# <a name="2-2"></a>
# ## Configuration
# [ToC](#ToC)  

configurations = [
    TransformerConfiguration(
        name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": True}, params_fitted={}
    ),
    TransformerConfiguration(
        name="ShiftByValueTransformer", fit=True, params_for_fit={"shift_value": -0.5}, params_fitted={}
    )
]

# <a name="2-3"></a>
# ## Transformations Example
# [ToC](#ToC)  


transformer = TransformationsExecutionerTransformer()
df_out = transformer.fit_predict(df_data.copy(), ATTRS, configurations)
df_out

df_inverse = transformer.inverse(df_out)
df_inverse

params = transformer.get_params()
params

transformer_restored = TransformationsExecutionerTransformer()
transformer_restored.restore_from_params(params)
df_out_from_restored = transformer_restored.predict(df_data.copy())
df_out_from_restored

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
