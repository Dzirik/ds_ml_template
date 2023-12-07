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

# # Machine Learning Models Documentation
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
#     - [Data](#2-1)   
#     - [Regression](#2-2)
#         - [Mean Model](#2-2-1)
#         - [Linear Regression](#2-2-2)
#             - [Normality of Residuals](#2-2-2-1)
#             - [Evaluation](#2-2-2-2)
#     - [Classification](#2-3)     
#         - [Max Entropy Classification Random Model](#2-3-1)
#             - [Not One Hot Encoding](#2-3-1-1)
#             - [One Hot Encoding](#2-3-1-2)
#         - [Distribution Classification Random Model](#2-3-2)
#             - [Not One Hot Encoding](#2-3-2-1)
#             - [One Hot Encoding](#2-3-2-2)        
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

from typing import Any
from numpy import float32, array, ndarray, dtype, double
from collections import Counter, OrderedDict
from pandas import Series

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

# +
from src.data.income_weather_data_generator import IncomeWeatherDataGenerator
from src.data.splitter import Splitter

from src.models.ml_r_mean_model import MeanModel
from src.models.ml_r_stats_lin_reg_model import StatsLinRegModel

from src.models.ml_c_max_entropy_classification_random_model import MaxEntropyClassificationRandomModel
from src.models.ml_c_distribution_classification_random_model import DistributionClassificationRandomModel

from src.visualisations.plotly_line_chart import PlotlyLineChart
from src.visualisations.plotly_histogram import PlotlyHistogram
from src.visualisations.plotly_histogram_multi import PlotlyHistogramMulti
from src.visualisations.plotly_bar_chart import PlotlyBarChart
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

# +
splitter = Splitter()

line_chart = PlotlyLineChart()
hist = PlotlyHistogram()
hist_multi = PlotlyHistogram()
bar_chart = PlotlyBarChart()
# -

# <a name="2-1"></a>
# ## Data
# [ToC](#ToC)  


(X_train, X_test), (Y_train_reg, Y_test_reg), (Y_train_bin, Y_test_bin), (Y_train_ter, Y_test_ter), \
(Y_train_ter_oh, Y_test_ter_oh) = IncomeWeatherDataGenerator().generate_basic_ml_data()

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


# <a name="2-2"></a>
# ## Regression
# [ToC](#ToC)  

def plot_histograms_and_residuals(train_res: ndarray[Any, dtype[double]], test_res: ndarray[Any, dtype[double]]) -> None:
    hist.plot(data=train_res.reshape(train_res.shape[0],), plot_title="Train Residuals", x_title="Y-Y hat")
    
    hist.plot(data=test_res.reshape(test_res.shape[0],), plot_title="Test Residuals", x_title="Y-Y hat")
    
    line_chart.plot(
        lines=[(array(range(train_res.shape[0])), train_res.reshape((train_res.shape[0],)))], 
        plot_title="Train Residuals", 
        x_title="Index", 
        y_title="Y-Y_hat"
    )
    
    line_chart.plot(
        lines=[(array(range(test_res.shape[0])), test_res.reshape((test_res.shape[0],)))], 
        plot_title="Test Residuals", 
        x_title="Index", 
        y_title="Y-Y_hat"
    )


# <a name="2-2-1"></a>
# ### Mean Model
# [ToC](#ToC) 

model = MeanModel()
model_params = {}

# +
train_predictions = model.fit_predict(X=X_train, Y=Y_train_reg, model_params=model_params)
test_predictions = model.predict(X=X_test)

train_residuals = model.get_residuals(X=X_train, Y=Y_train_reg)
test_residuals = model.get_residuals(X=X_test, Y=Y_test_reg)

print(train_predictions[0:5])
print(test_predictions[0:5])

print(f"Train R2 score: {model.get_r2_score(X=X_train, Y=Y_train_reg)}")
print(f"Test R2 score: {model.get_r2_score(X=X_test, Y=Y_test_reg)}")

plot_histograms_and_residuals(train_residuals, test_residuals)
# -

# <a name="2-2-2"></a>
# ### Linear Regression
# [ToC](#ToC) 
#
# [Parameters Explanation](https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a)
# - Omnibus value: 0 means perfect normalty  .
# - Prob(Omnibus): probability of residuals being normally distributed.
# - Durbin Watson: ideal is in between 1-2.
# - Prob(JB): The same as omnibus.


model = StatsLinRegModel()
model_params = {"intercept": True}

# +
train_predictions = model.fit_predict(X=X_train, Y=Y_train_reg, model_params=model_params)
test_predictions = model.predict(X=X_test)

train_residuals = model.get_residuals(X=X_train, Y=Y_train_reg)
test_residuals = model.get_residuals(X=X_test, Y=Y_test_reg)

print(train_predictions[0:5])
print(test_predictions[0:5])

print(f"Train R2 score: {model.get_r2_score(X=X_train, Y=Y_train_reg)}")
print(f"Test R2 score: {model.get_r2_score(X=X_test, Y=Y_test_reg)}")

plot_histograms_and_residuals(train_residuals, test_residuals)
# -

# <a name="2-2-2-1"></a>
# #### Normality of Residuals
# [ToC](#ToC) 
#
# Omnibus value: 0 means perfect normalty  
# Prob(Omnibus) probability of residuals being normally distributed.
#
# [Shapiro-Wilk Test](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)
# - Null hypothesis: x1, ..., xn is from a normally distributed population
# - [scipy.stats.shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
#
#
# [Kolmogorov Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
# - Null hypothesis: Disstribution is drawn from the tested distrubution.
# - [scipy.stats.kstest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html)
# - p > alpha - failed to reject H0, they are identical
# - p < alpha - rejecting H0, they are not identical

print(f"Stat, p-value: {model.do_normality_test_for_residuals(X_train, Y_train_reg, 'ks')}")
print(f"Stat, p-value: {model.do_normality_test_for_residuals(X_test, Y_test_reg, 'ks')}")

print(f"Stat, p-value: {model.do_normality_test_for_residuals(X_train, Y_train_reg, 'sw')}")
print(f"Stat, p-value: {model.do_normality_test_for_residuals(X_test, Y_test_reg, 'sw')}")

# <a name="2-2-2-2"></a>
# #### Evaluation
# [ToC](#ToC) 
#
# Results interpetation can be found [here](https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a).

lr_model = model.get_model()
model_stats = lr_model.params

model_stats

lr_model.summary()

summary = lr_model.summary()

summary.tables[0]

summary.tables[0].data

summary.tables[1]

summary.tables[2]

print(lr_model.summary())


# <a name="2-3"></a>
# ## Classification
# [ToC](#ToC) 

def get_ordered_classes_frequency(Y: ndarray[Any, dtype[double]]) -> OrderedDict:
    """
    Gets the frequncy of classes numbered order buy classe ids.
    :param Y: ndarray[Any, dtype[double]].
    """
    d = dict(Counter(Y.reshape((Y.shape[0],))))
    return OrderedDict(sorted(d.items()))


# <a name="2-3-1"></a>
# ### Max Entropy Classification Random Model
# [ToC](#ToC) 

# <a name="2-3-1-1"></a>
# #### Not One Hot Encoding
# [ToC](#ToC) 

model = MaxEntropyClassificationRandomModel()
model_params = {"oh": False}

# +
train_predictions = model.fit_predict(X=X_train, Y=Y_train_ter, model_params=model_params)
test_predictions = model.predict(X=X_test)
                                      
print(train_predictions[0:N_ROWS_TO_DISPLAY])
print(test_predictions[0:N_ROWS_TO_DISPLAY])
# -

print(get_ordered_classes_frequency(Y_train_ter))
print(get_ordered_classes_frequency(model.predict(X_train)))

# +
freq = get_ordered_classes_frequency(Y_train_ter)
bar_chart.plot(
    array_ids=array(list(freq.keys())), 
    array_values=array(list(freq.values())), 
    plot_title="",
    name_ids="",
    name_values="",
    order_by_values=False,
    reverse=False
)

freq = get_ordered_classes_frequency(model.predict(X_train))
bar_chart.plot(
    array_ids=array(list(freq.keys())), 
    array_values=array(list(freq.values())), 
    plot_title="",
    name_ids="",
    name_values="",
    order_by_values=False,
    reverse=False
)
# -

# <a name="2-3-1-2"></a>
# #### One Hot Encoding
# [ToC](#ToC) 

model = MaxEntropyClassificationRandomModel()
model_params = {"oh": True}

# +
train_predictions = model.fit_predict(X=X_train, Y=Y_train_ter_oh, model_params=model_params)
test_predictions = model.predict(X=X_test)
                                      
print(train_predictions[0:N_ROWS_TO_DISPLAY])
print(test_predictions[0:N_ROWS_TO_DISPLAY])
# -

print(Y_train_ter_oh.sum(axis=0))
print(train_predictions.sum(axis=0))

# +
freq = Y_train_ter_oh.sum(axis=0)
bar_chart.plot(
    array_ids=array(list(range(Y_train_ter_oh.shape[1]))), 
    array_values=array(freq), 
    plot_title="",
    name_ids="",
    name_values="",
    order_by_values=False,
    reverse=False
)

freq = model.predict(X_train).sum(axis=0)
bar_chart.plot(
    array_ids=array(list(range(Y_train_ter_oh.shape[1]))), 
    array_values=array(freq), 
    plot_title="",
    name_ids="",
    name_values="",
    order_by_values=False,
    reverse=False
)
# -

# <a name="2-3-2"></a>
# ### Distribution Classification Random Model
# [ToC](#ToC) 

# <a name="2-3-2-1"></a>
# #### Not One Hot Encoding
# [ToC](#ToC) 

model = DistributionClassificationRandomModel()
model_params = {"oh": False}

# +
train_predictions = model.fit_predict(X=X_train, Y=Y_train_ter, model_params=model_params)
test_predictions = model.predict(X=X_test)
                                      
print(train_predictions[0:N_ROWS_TO_DISPLAY])
print(test_predictions[0:N_ROWS_TO_DISPLAY])
# -

print(get_ordered_classes_frequency(Y_train_ter))
print(get_ordered_classes_frequency(model.predict(X_train)))

# +
freq = get_ordered_classes_frequency(Y_train_ter)
bar_chart.plot(
    array_ids=array(list(freq.keys())), 
    array_values=array(list(freq.values())), 
    plot_title="",
    name_ids="",
    name_values="",
    order_by_values=False,
    reverse=False
)

freq = get_ordered_classes_frequency(model.predict(X_train))
bar_chart.plot(
    array_ids=array(list(freq.keys())), 
    array_values=array(list(freq.values())), 
    plot_title="",
    name_ids="",
    name_values="",
    order_by_values=False,
    reverse=False
)
# -

# <a name="2-3-2-2"></a>
# #### One Hot Encoding
# [ToC](#ToC) 

model = DistributionClassificationRandomModel()
model_params = {"oh": True}

# +
train_predictions = model.fit_predict(X=X_train, Y=Y_train_ter_oh, model_params=model_params)
test_predictions = model.predict(X=X_test)
                                      
print(train_predictions[0:N_ROWS_TO_DISPLAY])
print(test_predictions[0:N_ROWS_TO_DISPLAY])
# -

print(Y_train_ter_oh.sum(axis=0))
print(train_predictions.sum(axis=0))

# +
freq = Y_train_ter_oh.sum(axis=0)
bar_chart.plot(
    array_ids=array(list(range(Y_train_ter_oh.shape[1]))), 
    array_values=array(freq), 
    plot_title="",
    name_ids="",
    name_values="",
    order_by_values=False,
    reverse=False
)

freq = model.predict(X_train).sum(axis=0)
bar_chart.plot(
    array_ids=array(list(range(Y_train_ter_oh.shape[1]))), 
    array_values=array(freq), 
    plot_title="",
    name_ids="",
    name_values="",
    order_by_values=False,
    reverse=False
)
# -

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
