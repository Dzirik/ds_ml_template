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

# # Income Weather Data Generator
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
#     - [Data Generation](#2-1) 
#     - [Explore Multidimensional Data](#2-2) 
#     - [Explore Basic Data](#2-3)    
#         - [Plot Output Variable](#2-3-1)
#         - [Not Transformed Data](#2-3-2)
#         - [Transformed Data](#2-3-3)
#     - [Data For Machine Learning](#2-4)
#         - [Basic Data](#2-4-1)
#         - [Multidimensional Data](#2-4-2)
# - [Final Timestamp](#3)

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC)

# Income data an artificial data set created for (time series) regression problems.
#
# There are two types of data there:
# - Basic one dimensional data.
# - More complex 2D data for DNN testing.
#
# NOTE:
# - The only attribute that would need normalisation is temperature. But the data is kept as it is, it should not affect 
# the output much, so it is not normalised.
#
# There are even whole train/test data sets for both catetory present there.
#
# All data:
# ```p
# data_gen = IncomeWeatherDataGenerator()
# (X_train, X_test), (Y_train_reg, Y_test_reg), (Y_train_bin, Y_test_bin), (Y_train_ter, Y_test_ter), \
# (Y_train_ter_oh, Y_test_ter_oh) = data_gen.generate_basic_ml_data()
#
# (X_train, X_test), (Y_train_reg, Y_test_reg), (Y_train_bin, Y_test_bin), (Y_train_ter, Y_test_ter), \
# (Y_train_ter_oh, Y_test_ter_oh) = data_gen.generate_multidim_ml_data()
# ```  
#
# Regression:
# ```p
# data_gen = IncomeWeatherDataGenerator()
# (X_train, X_test), (Y_train, Y_test), _, _, _ = data_gen.generate_basic_ml_data()
#
# (X_train, X_test), (Y_train, Y_test), _, _, _ = data_gen.generate_multidim_ml_data()
# ```  
#
# Binary Classification:
# ```p
# data_gen = IncomeWeatherDataGenerator()
# (X_train, X_test), _, (Y_train, Y_test), _, _ = data_gen.generate_basic_ml_data()
#
# (X_train, X_test), _, (Y_train, Y_test), _, _ = data_gen.generate_multidim_ml_data()
# ```  
#
# Tertiary Classification Dense:
# ```p
# data_gen = IncomeWeatherDataGenerator()
# (X_train, X_test), _, _, (Y_train, Y_test), _ = data_gen.generate_basic_ml_data()
#
# (X_train, X_test), _, _, (Y_train, Y_test), _ = data_gen.generate_multidim_ml_data()
# ```  
#
# Tertiary Classification One Hot:
# ```p
# data_gen = IncomeWeatherDataGenerator()
# (X_train, X_test), _, _, _, (Y_train, Y_test) = data_gen.generate_basic_ml_data()
#
# (X_train, X_test), _, _, _, (Y_train, Y_test) = data_gen.generate_multidim_ml_data()
# ```  

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

from pandas import options
from numpy import array
from collections import Counter

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)   
# Code, libraries, classes, functions from within the repository.

from src.data.income_weather_data_generator import IncomeWeatherDataGenerator, ATTR_OUTPUT, ATTR_DATE
from src.data.df_explorer import DFExplorer
from pprint import pprint
from src.visualisations.plotly_time_series import PlotlyTimeSeries

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
# ## Data Generation
# [ToC](#ToC)   
#
# The beta parameters follow the following order of attributes in encoded data frame:
# - TEMPERATURE
# - DAY_OF_WEEK_NUM_0
# - DAY_OF_WEEK_NUM_1
# - DAY_OF_WEEK_NUM_2
# - DAY_OF_WEEK_NUM_3
# - DAY_OF_WEEK_NUM_4
# - DAY_OF_WEEK_NUM_5
# - DAY_OF_WEEK_NUM_6
# - WEATHER_cloud
# - WEATHER_rain
# - WEATHER_sun
# - WEATHER_wind
#
# Column **RANDOM** is not used in generating. It is just random data to _confuse_ the regression and test importance.


start_date = "2018-01-01"
n = 40
betas = [30, 2, 1, 4, 3, 6, -1, -3, 0, -10, 25, 10]
sigma = 10

len(betas)

Logger().set_meantime("Generating Starts")
data_gen = IncomeWeatherDataGenerator()
df_data, df_data_transformed, X_multi, Y_multi = data_gen.generate(start_date, betas, n, sigma)
Logger().set_meantime("Generating Ends")

# <a name="2-2"></a>
# ## Explore Multidimensional Data
# [ToC](#ToC)   

attr_names = data_gen.get_attributes_names_multi()
print(len(attr_names))
print(attr_names)

data_gen.get_weights_multi()


def print_info(array_matrix: array) -> None:
    print(array_matrix.shape)
    print(array_matrix[0].shape)
    pprint(array_matrix[0])


print_info(X_multi)

print_info(Y_multi)

# <a name="2-3"></a>
# ## Explore Basic Data
# [ToC](#ToC)   

df_data.head()

df_data_transformed.shape

df_data_transformed.head()

df_explorer = DFExplorer()

# <a name="2-3-1"></a>
# ### Plot Output Variable
# [ToC](#ToC) 

# +
ts_visu = PlotlyTimeSeries()

ts = df_data[ATTR_OUTPUT]
ts.index = df_data[ATTR_DATE]
ts_visu.plot(
    series=[df_data[ATTR_OUTPUT]],
    plot_title="Weather Output Variable",
    y_title="Profit"
)
# -

# <a name="2-3-2"></a>
# ### Not Transformed Data
# [ToC](#ToC)   

df_explorer.print_info_about_data_frame(df=df_data)

df_explorer.print_attr_stats(df=df_data)

# <a name="2-3-3"></a>
# ### Transformed Data
# [ToC](#ToC)   

df_explorer.print_info_about_data_frame(df=df_data_transformed)

df_explorer.print_attr_stats(df=df_data_transformed)

# <a name="2-4"></a>
# ## Data For Machine Learning
# [ToC](#ToC)   

# <a name="2-4-1"></a>
# ### Basic Data
# [ToC](#ToC) 

# +
data_gen = IncomeWeatherDataGenerator()

(X_train, X_test), (Y_train_reg, Y_test_reg), (Y_train_bin, Y_test_bin), (Y_train_ter, Y_test_ter), \
(Y_train_ter_oh, Y_test_ter_oh) = data_gen.generate_basic_ml_data()
# -

print(X_train.shape)
print(Y_train_reg.shape)
print(Y_train_bin.shape)
print(Y_train_ter.shape)
print(Y_train_ter_oh.shape)
print("\n")
print(X_test.shape)
print(Y_test_reg.shape)
print(Y_test_bin.shape)
print(Y_test_ter.shape)
print(Y_test_ter_oh.shape)

print(Counter(list(Y_train_bin.reshape((Y_train_bin.shape[0],)).tolist())))
print(Counter(list(Y_train_ter.reshape((Y_train_ter.shape[0],)).tolist())))
print("\n")
print(Counter(list(Y_test_bin.reshape((Y_test_bin.shape[0],)).tolist())))
print(Counter(list(Y_test_ter.reshape((Y_test_ter.shape[0],)).tolist())))

# <a name="2-4-2"></a>
# ### Multidimensional Data
# [ToC](#ToC) 

# +
data_gen = IncomeWeatherDataGenerator()

(X_train, X_test), (Y_train_reg, Y_test_reg), (Y_train_bin, Y_test_bin), (Y_train_ter, Y_test_ter), \
(Y_train_ter_oh, Y_test_ter_oh) = data_gen.generate_multidim_ml_data()
# -

print(X_train.shape)
print(Y_train_reg.shape)
print(Y_train_bin.shape)
print(Y_train_ter.shape)
print(Y_train_ter_oh.shape)
print("\n")
print(X_test.shape)
print(Y_test_reg.shape)
print(Y_test_bin.shape)
print(Y_test_ter.shape)
print(Y_test_ter_oh.shape)

print(Counter(list(Y_train_bin.reshape((Y_train_bin.shape[0],)).tolist())))
print(Counter(list(Y_train_ter.reshape((Y_train_ter.shape[0],)).tolist())))
print("\n")
print(Counter(list(Y_test_bin.reshape((Y_test_bin.shape[0],)).tolist())))
print(Counter(list(Y_test_ter.reshape((Y_test_ter.shape[0],)).tolist())))

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)   

Logger().end_timer()
