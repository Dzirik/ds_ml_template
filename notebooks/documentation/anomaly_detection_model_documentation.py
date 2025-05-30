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

# # Anomaly Detection Model Documentation
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
#     - [Data](#2-1) 
#     - [Parameters](#2-2)
#     - [Model](#2-3)
#         - [Fit and Predict](#2-3-1)
#         - [FitPredict](#2-3-2) 
#         - [Reconstruct](#2-3-3)
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
from numpy import array

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

from src.utils.date_time_functions import create_datetime_id
from src.models.anomaly_detection_rules_model import AnomalyDetectionParams, AnomalyDetectionRulesModel, DetectionRuleParams

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



# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)  

# <a name="2-1"></a>
# ## Data
# [ToC](#ToC)  

data_fit = array([1., 2.]).reshape((2,1))
data_fit

data_predict = array([
    range(8),
    range(1, 9),
    8 * [-0.2],
    8 * [0.],
    8 * [3.]
])
data_predict

# <a name="2-2"></a>
# ## Parameters
# [ToC](#ToC)  


params = AnomalyDetectionParams(
    rules_definition=[
        DetectionRuleParams(
            name="FirstWecoRule",
            params={"side": "both", "alpha": 1}
        ),
        DetectionRuleParams(
            name="SecondWecoRule",
            params={"side": "both", "alpha": 1.}
        ),
        DetectionRuleParams(
            name="ThirdWecoRule",
            params={"side": "both", "alpha": 1.}
        ),
        DetectionRuleParams(
            name="FourthWecoRule",
            params={"side": "both", "alpha": 1.}
        )
    ]
)

# <a name="2-3"></a>
# ## Model
# [ToC](#ToC)  

# <a name="2-3-1"></a>
# ### Fit and Predict
# [ToC](#ToC)  

# +
model = AnomalyDetectionRulesModel()

model.fit(data_fit, params)
out = model.predict(data_predict)
print(model.get_rule_names())
print(out)
# -

# <a name="2-3-2"></a>
# ### FitPredict
# [ToC](#ToC)  

# +
model = AnomalyDetectionRulesModel()

out = model.fit_predict(data_fit, data_predict, params)
print(model.get_rule_names())
print(out)
# -

# <a name="2-3-3"></a>
# ### Reconstruct
# [ToC](#ToC)

# +
model = AnomalyDetectionRulesModel()

out = model.fit_predict(data_fit, data_predict, params)
print(model.get_rule_names())
print(out)

model_rec = AnomalyDetectionRulesModel()
model_rec.restore_from_params(model.get_params())
out = model_rec.predict(data_predict)
print(model_rec.get_rule_names())
print(out)
# -

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
