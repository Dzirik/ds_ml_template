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

# # Transformations Executioner Pipeline Documentation
# *Version:* `1.1` *(Jupytext, time measurements, logger, param notebook execution)*

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
#     - [Atomic Transformers Configurations](#2-1) 
#     - [Fit and Predict](#2-2)
#         - [Simple Example](#2-2-1)   
#             - [Data](#2-2-1-1)
#             - [Configuration](#2-2-1-2)
#             - [Pipeline](#2-2-1-3)
#         - [Complex Example](#2-2-2)
#             - [Data](#2-2-2-1)
#             - [Configuration](#2-2-2-2)
#             - [Pipeline](#2-2-2-3)
#         - [Complex Example From Fitted](#2-2-3)
#             - [Data](#2-2-3-1)
#             - [Configuration](#2-2-3-2)
#             - [Pipeline](#2-2-3-3)
#         - [Complex Example From Saved and Loaded Config](#2-2-4)
#             - [Data](#2-2-4-1)
#             - [Configuration](#2-2-4-2)
#             - [Pipeline](#2-2-4-3)
#         - [Inverse](#2-2-5)
#     - [Execute](#2-3)
#         - [Fit Predict Mode](#2-3-1)
#             - [Data](#2-3-1-1)
#             - [Configuration](#2-3-1-2)
#             - [Pipeline](#2-3-1-3)
#         - [Predict Model](#2-3-2)
#             - [Data](#2-3-2-1)
#             - [Configuration](#2-3-2-2)
#             - [Pipeline](#2-3-2-3)
#         - [Inverse](#2-3-3)
#         - [Comparison](#2-3-4)
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

# > Constants for overall behaviour.

LOGGER_CONFIG_NAME = "logger_file_console" # default
PYTHON_CONFIG_NAME = "python_personal" # default
CREATE_BUTTON = False
ADDAPT_WIDTH = False

options.display.max_rows = 500
options.display.max_columns = 500
envs = Envs()
envs.set_logger(LOGGER_CONFIG_NAME)
envs.set_config(PYTHON_CONFIG_NAME)
Logger().start_timer(f"NOTEBOOK; Notebook name: {NOTEBOOK_NAME}")
if SUPPORT_FUNCTIONS_READ and CREATE_BUTTON:
    create_button()
if ADDAPT_WIDTH:
    display(HTML("<style>.container { width:100% !important; }</style>")) # notebook width

# <a name="1-3"></a>
# ### External Libraries
# [ToC](#ToC)  



# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)  
# Code, libraries, classes, functions from within the repository.

# +
from src.data.saver_and_loader import SaverAndLoader

from src.data.anomaly_detection_sample_data import AnomalyDetectionSampleData, ATTRS, ATTR_DATE_TIME, ATTR_ID, plot_data_frame_series

from src.transformations.transformations_executioner_transformer import TransformationsExecutionerTransformer, TransformerConfiguration
from src.pipelines.transfomations_executioner_pipeline_config_data import TransformationExecutionerConfiguration, \
    TransformationsExecutionerPipelineConfigData
from src.pipelines.transfomations_executioner_pipeline import TransformationsExecutionerlPipeline

from src.constants.global_constants import FP, P
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

# <a name="2-1"></a>
# ## Atomic Transformers Configurations
# [ToC](#ToC)  


# ```p
# C_DIFF = TransformerConfiguration(
#     name="DifferenceTransformer", fit=True, params_for_fit={"periods": 1}, params_fitted={}
# )
# C_DIFF_PERC = TransformerConfiguration(
#     name="DifferencePercentageChangeTransformer", fit=True, params_for_fit={"periods": 1}, params_fitted={}
# )
# C_LOG = TransformerConfiguration(
#     name="LogTransformer", fit=True, params_for_fit={"base": 10.}, params_fitted={}
# )
# C_LOG_FITTED = TransformerConfiguration(
#     name="LogTransformer", fit=False, params_for_fit={}, params_fitted={"base": 10.}
# )
# C_MM_MULTI = TransformerConfiguration(
#     name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": False}, params_fitted={}
# )
# C_MM_SINGLE = TransformerConfiguration(
#     name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": True}, params_fitted={}
# )
# C_Q = TransformerConfiguration(
#     name="QuantileTransformer", fit=True, params_for_fit={}, params_fitted={}
# )
# C_SHIFT = TransformerConfiguration(
#     name="ShiftByValueTransformer", fit=True, params_for_fit={"shift_value": -0.5}, params_fitted={}
# )
# ```

# <a name="2-2"></a>
# ## Fit and Predict
# [ToC](#ToC) 

# <a name="2-2-1"></a>
# ### Simple Example
# [ToC](#ToC)  

# <a name="2-2-1-1"></a>
# #### Data
# [ToC](#ToC)

# +
dfs, df_whole = AnomalyDetectionSampleData(abs_values=False, add_zeros=False).create_list_of_data_frames(n=10, seed_number=398)

dfs
# -

# <a name="2-2-1-2"></a>
# #### Configuration
# [ToC](#ToC)

C_MM_MULTI = TransformerConfiguration(
    name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": False}, params_fitted={}
)
C_SHIFT = TransformerConfiguration(
    name="ShiftByValueTransformer", fit=True, params_for_fit={"shift_value": -0.5}, params_fitted={}
)

config_data = TransformationsExecutionerPipelineConfigData(
    name="test",
    trans_executioners_confs=[
        TransformationExecutionerConfiguration(
            create=True,
            attrs=ATTRS,
            configurations=[
                C_MM_MULTI, C_SHIFT
            ]
        )                                               
    ]
)

# <a name="2-2-1-3"></a>
# #### Pipeline
# [ToC](#ToC)

# +
config_file_name = None

pipeline = TransformationsExecutionerlPipeline(config_file_name)

pipeline.set_config_data(config_data)
pipeline.fit(dfs)

df_out = pipeline.predict(dfs)
df_out_concatenated = pipeline._concatenate_data_frames(df_out)

df_out_whole = pipeline.predict([df_whole])

assert df_out_whole[0].equals(df_out_concatenated)

df_out_concatenated
# -

# <a name="2-2-2"></a>
# ### Complex Example
# [ToC](#ToC)  

# <a name="2-2-2-1"></a>
# #### Data
# [ToC](#ToC)

# +
dfs, df_whole = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).create_list_of_data_frames(n=10, seed_number=398)

dfs
# -

# <a name="2-2-2-2"></a>
# #### Configuration
# [ToC](#ToC)

C_LOG = TransformerConfiguration(
    name="LogTransformer", fit=True, params_for_fit={"base": 10.}, params_fitted={}
)
C_MM_MULTI = TransformerConfiguration(
    name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": False}, params_fitted={}
)
C_SHIFT = TransformerConfiguration(
    name="ShiftByValueTransformer", fit=True, params_for_fit={"shift_value": -0.5}, params_fitted={}
)

config_data = TransformationsExecutionerPipelineConfigData(
    name="test",
    trans_executioners_confs=[
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[0], ATTRS[1]],
            configurations=[
                C_MM_MULTI, C_SHIFT
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[2]],
            configurations=[
                C_SHIFT
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[3], ATTRS[4]],
            configurations=[
                C_LOG
            ]
        )
    ]
)

pipeline._exec_trs[0]._transformers[0].get_params()

# <a name="2-2-2-3"></a>
# #### Pipeline
# [ToC](#ToC)

# +
config_file_name = None

pipeline = TransformationsExecutionerlPipeline(config_file_name)

pipeline.set_config_data(config_data)
pipeline.fit(dfs)

df_out = pipeline.predict(dfs)
df_out_concatenated = pipeline._concatenate_data_frames(df_out)

df_out_whole = pipeline.predict([df_whole])

assert df_out_whole[0].equals(df_out_concatenated)

df_out_concatenated
# -

df_out_with_fit = df_out

# <a name="2-2-3"></a>
# ### Complex Example From Fitted
# [ToC](#ToC)  

# <a name="2-2-3-1"></a>
# #### Data
# [ToC](#ToC)

# +
dfs, df_whole = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).create_list_of_data_frames(n=10, seed_number=398)

dfs
# -

# <a name="2-2-3-2"></a>
# #### Configuration
# [ToC](#ToC)

C_LOG_FITTED = TransformerConfiguration(
    name="LogTransformer", fit=False, params_for_fit={}, params_fitted={"base": 10.}
)
C_MM_MULTI_FITTED = TransformerConfiguration(
    name="MinMaxScalingTransformer", fit=False, params_for_fit={}, params_fitted={
        'main_params': {'clip': False, 'copy': True, 'feature_range': (0.0, 1.0)},
        'min_': [0.0, 0.0],
        'scale_': [0.21979294, 0.21979294],
        'data_min_': [0.0, 0.0],
        'data_max_': [4.5497367, 4.5497367]
    }
)
C_SHIFT_FITTED = TransformerConfiguration(
    name="ShiftByValueTransformer", fit=False, params_for_fit={}, params_fitted={"shift_value": -0.5}
)

config_data_fitted = TransformationsExecutionerPipelineConfigData(
    name="test",
    trans_executioners_confs=[
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[0], ATTRS[1]],
            configurations=[
                C_MM_MULTI_FITTED, C_SHIFT_FITTED
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[2]],
            configurations=[
                C_SHIFT_FITTED
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[3], ATTRS[4]],
            configurations=[
                C_LOG_FITTED
            ]
        )
    ]
)

# <a name="2-2-3-3"></a>
# #### Pipeline
# [ToC](#ToC)

# +
config_file_name = None

pipeline = TransformationsExecutionerlPipeline(config_file_name)

pipeline.set_config_data(config_data_fitted)
pipeline.fit(dfs)

df_out = pipeline.predict(dfs)
df_out_concatenated = pipeline._concatenate_data_frames(df_out)

df_out_whole = pipeline.predict([df_whole])

assert df_out_whole[0].equals(df_out_concatenated)

df_out_concatenated
# -

df_out_from_fitted = df_out

for df_1, df_2 in zip(df_out_with_fit, df_out_from_fitted):
    assert df_2.equals(df_2)

# <a name="2-2-4"></a>
# ### Complex Example Restoration
# [ToC](#ToC)  

# <a name="2-2-4-1"></a>
# #### Data
# [ToC](#ToC)

# +
dfs, df_whole = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).create_list_of_data_frames(n=10, seed_number=398)

dfs
# -

# <a name="2-2-4-2"></a>
# #### Configuration
# [ToC](#ToC)

C_LOG = TransformerConfiguration(
    name="LogTransformer", fit=True, params_for_fit={"base": 10.}, params_fitted={}
)
C_MM_MULTI = TransformerConfiguration(
    name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": False}, params_fitted={}
)
C_SHIFT = TransformerConfiguration(
    name="ShiftByValueTransformer", fit=True, params_for_fit={"shift_value": -0.5}, params_fitted={}
)

config_data = TransformationsExecutionerPipelineConfigData(
    name="test",
    trans_executioners_confs=[
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[0], ATTRS[1]],
            configurations=[
                C_MM_MULTI, C_SHIFT
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[2]],
            configurations=[
                C_SHIFT
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[3], ATTRS[4]],
            configurations=[
                C_LOG
            ]
        )
    ]
)

# <a name="2-2-4-3"></a>
# #### Pipeline
# [ToC](#ToC)

# +
config_file_name = None

pipeline = TransformationsExecutionerlPipeline(config_file_name)

pipeline.set_config_data(config_data)
pipeline.fit(dfs)

df_out = pipeline.predict(dfs)
df_out_concatenated = pipeline._concatenate_data_frames(df_out)

df_out_whole = pipeline.predict([df_whole])

assert df_out_whole[0].equals(df_out_concatenated)

df_out_concatenated
# -

df_out_from_before_restoration = df_out

config_data_fit = pipeline.get_config_data()
config_data_fit

# +
config_file_name = None

pipeline_2 = TransformationsExecutionerlPipeline(config_file_name)

pipeline_2.set_config_data(config_data_fit)
pipeline_2.fit(dfs)

df_out = pipeline_2.predict(dfs)
df_out_concatenated = pipeline_2._concatenate_data_frames(df_out)

df_out_whole = pipeline_2.predict([df_whole])

assert df_out_whole[0].equals(df_out_concatenated)

df_out_concatenated
# -

df_out_from_after_restoration = df_out_whole

for df_1, df_2 in zip(df_out_with_fit, df_out_from_fitted):
    assert df_2.equals(df_2)

# <a name="2-2-5"></a>
# ### Inverse
# [ToC](#ToC)  

# **NOT DONE YET**

# <a name="2-3"></a>
# ## Execute
# [ToC](#ToC) 

# <a name="2-3-1"></a>
# ### Fit Predict Mode
# [ToC](#ToC)  

# <a name="2-3-1-1"></a>
# #### Data
# [ToC](#ToC)

# +
dfs, df_whole = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).create_list_of_data_frames(n=10, seed_number=398)

dfs
# -

# <a name="2-3-1-2"></a>
# #### Configuration
# [ToC](#ToC)

C_LOG = TransformerConfiguration(
    name="LogTransformer", fit=True, params_for_fit={"base": 10.}, params_fitted={}
)
C_MM_MULTI = TransformerConfiguration(
    name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": False}, params_fitted={}
)
C_SHIFT = TransformerConfiguration(
    name="ShiftByValueTransformer", fit=True, params_for_fit={"shift_value": -0.5}, params_fitted={}
)

config_data = TransformationsExecutionerPipelineConfigData(
    name="test",
    trans_executioners_confs=[
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[0], ATTRS[1]],
            configurations=[
                C_MM_MULTI, C_SHIFT
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[2]],
            configurations=[
                C_SHIFT
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[3], ATTRS[4]],
            configurations=[
                C_LOG
            ]
        )
    ]
)

# <a name="2-3-1-3"></a>
# #### Pipeline
# [ToC](#ToC)

# +
config_file_name = None

pipeline = TransformationsExecutionerlPipeline(config_file_name)

pipeline.set_config_data(config_data)
dfs_train_out, dfs_test_out = pipeline.execute(dfs_train=dfs, dfs_test=dfs, type_of_method=FP)
# -

# <a name="2-3-2"></a>
# ### Predict Mode
# [ToC](#ToC)  

# <a name="2-3-2-1"></a>
# #### Data
# [ToC](#ToC)

# +
dfs, df_whole = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).create_list_of_data_frames(n=10, seed_number=398)

dfs
# -

# <a name="2-3-2-2"></a>
# #### Configuration
# [ToC](#ToC)

C_LOG_FITTED = TransformerConfiguration(
    name="LogTransformer", fit=False, params_for_fit={}, params_fitted={"base": 10.}
)
C_MM_MULTI_FITTED = TransformerConfiguration(
    name="MinMaxScalingTransformer", fit=False, params_for_fit={}, params_fitted={
        'main_params': {'clip': False, 'copy': True, 'feature_range': (0.0, 1.0)},
        'min_': [0.0, 0.0],
        'scale_': [0.21979294, 0.21979294],
        'data_min_': [0.0, 0.0],
        'data_max_': [4.5497367, 4.5497367]
    }
)
C_SHIFT_FITTED = TransformerConfiguration(
    name="ShiftByValueTransformer", fit=False, params_for_fit={}, params_fitted={"shift_value": -0.5}
)

config_data_fitted = TransformationsExecutionerPipelineConfigData(
    name="test",
    trans_executioners_confs=[
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[0], ATTRS[1]],
            configurations=[
                C_MM_MULTI_FITTED, C_SHIFT_FITTED
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[2]],
            configurations=[
                C_SHIFT_FITTED
            ]
        ),
        TransformationExecutionerConfiguration(
            create=True,
            attrs=[ATTRS[3], ATTRS[4]],
            configurations=[
                C_LOG_FITTED
            ]
        )
    ]
)

# <a name="2-3-2-3"></a>
# #### Pipeline
# [ToC](#ToC)

# +
config_file_name = None

pipeline = TransformationsExecutionerlPipeline(config_file_name)

pipeline.set_config_data(config_data_fitted)
_, dfs_test_out_fitted = pipeline.execute(dfs_train=None, dfs_test=dfs, type_of_method=P)
# -

# <a name="2-3-3"></a>
# ### Inverse
# [ToC](#ToC)  

# **NOT DONE YET**

# <a name="2-3-Q4"></a>
# ### Comparison
# [ToC](#ToC)  

# +
for df_1, df_2 in zip(dfs_train_out, dfs_test_out_fitted):
    assert df_2.equals(df_2)
    
for df_1, df_2 in zip(dfs_test_out, dfs_test_out_fitted):
    assert df_2.equals(df_2)
# -

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)  

Logger().end_timer()
