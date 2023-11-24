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

# # Time Series Cross Validation Documentation
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
#     - [Inputs](#2-1) 
#         - [Income Weather Data](#2-1-1)  
#         - [Sample Data](#2-1-2)  
#         - [Time (date_time)](#2-1-3)    
#         - [Split Dates (split_dates)](#2-1-4)  
#         - [Data (data)](#2-1-5)  
#     - [Basic Use Cases](#2-2)  
#         - [No Split Date](#2-2-1)
#         - [Split Dates](#2-2-2) 
#     - [Types of Cross Validation](#2-3)
#         - [One Fold Only](#2-3-1)  
#         - [Train Test Split](#2-3-2)  
#         - [Train Dev Test Split](#2-3-3) 
#         - [K-Fold Cross Validation](#2-3-4)  
#     - [Shuffling](#2-4)  
#         - [No Split](#2-4-1)  
#         - [Train Test Split](#2-4-2)  
#         - [K Fold Case](#2-4-3)  
#     - [Extract Arrays from List of Data Frames](#2-5)
#     - [Business and Production Testing](#2-6)  
# - [Final Timestamp](#3)

# <a name="0"></a>
# # Notebook Description
# [ToC](#ToC)

# Notebook for documentation and testing of cross validation code for time series.

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

LOGGER_CONFIG_NAME = "logger_file" # default: logger_file_console
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

from pprint import pprint
from datetime import datetime, timedelta
from pandas import DatetimeIndex
from numpy import array, concatenate
from pandas._libs.tslibs.timestamps import Timestamp
from sklearn.utils import shuffle
from pandas import DataFrame

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)   
# Code, libraries, classes, functions from within the repository.

# +
from src.data.income_weather_data_generator import IncomeWeatherDataGenerator, ATTR_DATE, ATTR_TEMPERATURE, ATTR_RANDOM, ATTR_OUTPUT
from src.data.df_explorer import DFExplorer

# from src.transformations.time_series_windows import TimeSeriesWindowsNumpy
from src.transformations.time_series_cv import TimeSeriesCV, create_weather_data, create_sample_data
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
DATA_PROCESSING_CONFIG_NAME = "data_processing_basic"

# #### Constants for Setting Automatic Run
# [ToC](#ToC)   



# #### Notebook Specific Constants
# [ToC](#ToC)   



# <a name="2"></a>
# # ANALYSIS
# [ToC](#ToC)   

 config_data = Config().get_data()

# <a name="2-1"></a>
# ## Inputs
# [ToC](#ToC)   
#
# This is a bit tricky part, because dealing with time. I tested multiple scenarios consistend with the main data format I use for time series - DatetimeIndex.
#
# Because it is part of ML, numpy, pipeline, the data has to be in array format. On the contrary, the timestamps of respective observations stay in the DatetimeIndex format and then in the format of EDA & FE pipeline, pandas. This was the best idea I had how to do it, because:
# - DatetimeIndex is the format I use for handling dates and times.
# - For creating the window splits, the numpy solution is far most the fastest (please see the documentatino of *time_series_windows*).
#
# Short description:
#
# - date_time: DatetimeIndex. Timestamp corresponding to the data points.
# - split_dates. Union[datetime.datetime, pandas._libs.tslibs.timestamps.Timestamp]. Dates for splits in the date_time. EDGES ARE ADDED AUTOMATICALLY. This is not very good to mix format, but both work. So I keep them both in tests.
# - data: array. At least two dimensional array. First dimension has to match the length of date_time variable.


# <a name="2-1-1"></a>
# ### Income Weather Data
# [ToC](#ToC)   
#
# This is a sample data for my usage, please see the separate documentation notebook for that.
#
# It reflects my standard time formatting, so I have it here for testing.

w_date_time, w_split_dates, w_X_data, w_Y_data = create_weather_data()

# <a name="2-1-2"></a>
# ### Sample Data
# [ToC](#ToC)   
#
# Sample data set for tests.

s_date_time, s_split_dates, s_X_data, s_Y_data, s_date_time_not_sorted = create_sample_data(31)

# <a name="2-1-3"></a>
# ### Time (date_time)
# [ToC](#ToC)   


w_date_time

s_date_time

s_date_time_not_sorted

# <a name="2-1-4"></a>
# ### Split Dates (split_dates)
# [ToC](#ToC)   

w_split_dates

s_split_dates

# <a name="2-1-5"></a>
# ### Data (data)
# [ToC](#ToC)   

w_X_data

w_Y_data

s_X_data

s_Y_data

# <a name="2-2"></a>
# ## Basic Use Cases
# [ToC](#ToC) 
#
# The output format is `List[((X_train, Y_train), (X_test, Y_test))]`

transformer = TimeSeriesCV()

# <a name="2-2-1"></a>
# ### No Split Date
# [ToC](#ToC) 
#
# The basic use case is about not having any split of the data, so taking the data as one. In this case, **no test data is created**, only train data with windows depending on windows settings.
#
# There are more cases in this set up:
# - `X_data`, `Y_data`
#     - They are both equal. That it is simply splitting past (X) and future (Y).
#     - They are not equal. In this case, Y should be a subset of attributes of X. Then the past is full and the future is only from that subset.
# - `Y_same_observation`: 
#     - If `True`, the first observation of Y is from the same date and time as the last observation of X.
#     - If `False`, the first observation of Y is time-wise the following to the X.

# #### No Split Date
# [ToC](#ToC) 

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 10
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = []
X_data = X_data
Y_data = X_data
Y_same_observation = True
input_window_len = 1
output_window_len = 1
shift = 1
shuffle_data = False

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)
pprint(cv)

# +
n = 10
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = []
X_data = X_data
Y_data = X_data
Y_same_observation = False
input_window_len = 1
output_window_len = 1
shift = 1
shuffle_data = False

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)
pprint(cv)

# +
n = 10
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = []
X_data = X_data
Y_data = Y_data
Y_same_observation = False
input_window_len = 1
output_window_len = 1
shift = 1
shuffle_data = False

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)
pprint(cv)
# -

# <a name="2-2-2"></a>
# ### Split Dates
# [ToC](#ToC) 
#
# Beginning and end of the interval is added automatically.

s_split_dates

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
date_time = s_date_time
split_dates = s_split_dates
X_data = s_X_data
Y_data = s_Y_data
Y_same_observation = False
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = False

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)
pprint(cv)
# -

# <a name="2-3"></a>
# ## Types of Cross Validation
# [ToC](#ToC) 
#
# The code is designed to cover all main cases.

# <a name="2-3-1"></a>
# ### One Fold Only
# [ToC](#ToC) 
#
# In this case all possible cases across the data are created.

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 10
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = []
X_data = X_data
Y_data = Y_data
Y_same_observation = True
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = False

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)
pprint(cv)
# -

# <a name="2-3-2"></a>
# ### Train Test Split
# [ToC](#ToC) 
#
# - Add one timestamp into split_dates in which the split should be made.
# - Getting train, test split:
#     - First possibility is to take only first element of resulted cv variable.
#     - Second possibility is to take only splits for respective intervals instead of cv variable.

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 11
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = [date_time[6]]
X_data = X_data
Y_data = Y_data
Y_same_observation = True
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = False

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)
((X_train, Y_train), (X_test, Y_test)) = cv[0]

print("### TRAIN ###")
pprint(X_train)
pprint(Y_train)
print("### TEST ###")
pprint(X_test)
pprint(Y_test)

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 11
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = [date_time[6]]
X_data = X_data
Y_data = Y_data
Y_same_observation = True
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = False

_ = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)

([X_train, X_test], [Y_train, Y_test]) = transformer.get_splits()

print("### TRAIN ###")
pprint(X_train)
pprint(Y_train)
print("### TEST ###")
pprint(X_test)
pprint(Y_test)
# -

# <a name="2-3-3"></a>
# ### Train Dev Test Split
# [ToC](#ToC) 
#
# - Add two timestamps into split_dates in which the splits should be made.
# - The train/dev/test splits are then taken out of get_splits() method.

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 15
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = [date_time[5], date_time[10]]
X_data = X_data
Y_data = Y_data
Y_same_observation = True
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = False

_ = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)

([X_train, X_dev, X_test], [Y_train, Y_dev, Y_test]) = transformer.get_splits()

print("### TRAIN ###")
pprint(X_train)
pprint(Y_train)
print("### DEV ###")
pprint(X_dev)
pprint(Y_dev)
print("### TEST ###")
pprint(X_test)
pprint(Y_test)
# -

# <a name="2-3-4"></a>
# ### K-Fold Cross Validation
# [ToC](#ToC) 
#
# - Add as many timestamps into split_dates in which the splits should be made as you want.
# - The split is in cv variable in the structure `List[((X_rest, Y_rest), (X_last, Y_last)), ...]`

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 20
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = [date_time[5], date_time[10], date_time[15]]
X_data = X_data
Y_data = Y_data
Y_same_observation = True
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = False

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)

((X_train, Y_train), (X_test, Y_test)) = cv[0]

print("### TRAIN ###")
pprint(X_train)
pprint(Y_train)
print("### TEST ###")
pprint(X_test)
pprint(Y_test)
# -

pprint(cv)

# <a name="2-4"></a>
# ## Shuffling
# [ToC](#ToC) 

# <a name="2-4-1"></a>
# ### No Split
# [ToC](#ToC) 

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 10
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = []
X_data = X_data
Y_data = X_data
Y_same_observation = True
input_window_len = 1
output_window_len = 1
shift = 1
shuffle_data = True

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)
pprint(cv)
# -

# <a name="2-4-2"></a>
# ### Train Test Split
# [ToC](#ToC) 

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 20
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = [date_time[10]]
X_data = X_data
Y_data = Y_data
Y_same_observation = True
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = True

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)
((X_train, Y_train), (X_test, Y_test)) = cv[0]

print("### TRAIN ###")
pprint(X_train)
pprint(Y_train)
print("### TEST ###")
pprint(X_test)
pprint(Y_test)

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 20
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = [date_time[10]]
X_data = X_data
Y_data = Y_data
Y_same_observation = True
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = True

_ = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)

([X_train, X_test], [Y_train, Y_test]) = transformer.get_splits()

print("### TRAIN ###")
pprint(X_train)
pprint(Y_train)
print("### TEST ###")
pprint(X_test)
pprint(Y_test)
# -

# <a name="2-4-3"></a>
# ### K Fold Case
# [ToC](#ToC) 

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 20
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = [date_time[5], date_time[10], date_time[15]]
X_data = X_data
Y_data = Y_data
Y_same_observation = True
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = True

cv = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)

pprint(cv)
# -

# <a name="2-5"></a>
# ## Extract Arrays from List of Data Frames
# [ToC](#ToC) 
#
# In this case, a list of data frames is given for together with attributes for X variable and Y variable and the goal is to create windows for all single data frames and then concatete them into one.

from src.transformations.time_series_cv import create_windows_from_list_of_data_frames

attr_date_time = "DATETIME"
attrs_X = ["A1", "A2", "A3"]
attrs_Y = ["OUTPUT"]

df_1 = DataFrame()
df_1[attr_date_time] = date_time[0:10]
df_1[attrs_X[0]] = X_data[0:10, 0]
df_1[attrs_X[1]] = X_data[0:10, 1]
df_1[attrs_X[2]] = [1.] * 10
df_1[attrs_Y[0]] = Y_data[0:10, 0]
df_1

df_2 = DataFrame()
df_2[attr_date_time] = date_time[10:]
df_2[attrs_X[0]] = X_data[10:, 0]
df_2[attrs_X[1]] = X_data[10:, 0]
df_2[attrs_X[2]] = [10.] * 10
df_2[attrs_Y[0]] = Y_data[10:, 0]
df_2

X, Y = create_windows_from_list_of_data_frames(
    dfs=[df_1, df_2], 
    attr_date_time=attr_date_time, 
    attrs_X=attrs_X, 
    attrs_Y=attrs_Y, 
    y_same_observation=True,
    input_window_len=8,
    output_window_len=1,
    shift=1,
    shuffle_data=False,
    shuffle_seed=None
)

print(X.shape)
print(Y.shape)

X

Y

# <a name="2-6"></a>
# ## Business and Production Testing
# [ToC](#ToC)  
#
# For business and production testing, splits with chronological windows are needed too. For given split dates, the `get_splits()` method is used.

# +
import src.transformations.time_series_cv as T
from importlib import reload

reload(T)

transformer = T.TimeSeriesCV()

# +
n = 20
date_time, split_dates, X_data, Y_data, _ = create_sample_data(n)

date_time = date_time
split_dates = [date_time[5], date_time[10], date_time[15]]
X_data = X_data
Y_data = Y_data
Y_same_observation = True
input_window_len = 3
output_window_len = 2
shift = 1
shuffle_data = False

_ = transformer.fit_predict(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, 
                             output_window_len, shift, shuffle_data)

X_splits, Y_splits = transformer.get_splits()
# -

for i, (X, Y) in enumerate(zip(X_splits, Y_splits)):
    print(f"Split {i}")
    print(X)
    print(Y)
    print("\n")

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)   

Logger().end_timer()
