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

# # Tensorflow Installation and Testing
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
#     - [Overview of Devices Available](#2-1)   
#     - [Another GPU Availability Tests](#2-2) 
#     - [Get GPU Names](#2-3)  
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

# +
from pprint import pprint

from tensorflow.config.experimental import list_physical_devices
from tensorflow.python.client.device_lib import list_local_devices
from tensorflow.test import is_gpu_available
# -

# <a name="1-4"></a>
# ### Internal Code
# [ToC](#ToC)   
# Code, libraries, classes, functions from within the repository.



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

config = Config().get()

# <a name="2-1"></a>
# ## Overview of Devices Available
# [ToC](#ToC)   
#
# Results from computer with GPU:
#
#     Device Name: /device:CPU:0
#     Device Description: 
#
#
#     Device Name: /device:XLA_CPU:0
#     Device Description: device: XLA_CPU device
#
#
#     Device Name: /device:GPU:0
#     Device Description: device: 0, name: GeForce RTX 3090, pci bus id: 0000:0b:00.0, compute capability: 8.6
#
#
#     Device Name: /device:XLA_GPU:0
#     Device Description: device: XLA_GPU device
#     
# Results from computer without GPU:
#
#     Device Name: /device:CPU:0
#     Device Description: 
#
#
#     Device Name: /device:XLA_CPU:0
#     Device Description: device: XLA_CPU device


device_list = list_local_devices()
for device in device_list:
    print(f"Device Name: {device.name}")
    print(f"Device Description: {device.physical_device_desc}")
    print("\n")


def get_gpu_devices() -> str:
    """
    Returns name of the GPU devices available.
    """
    device_list = list_local_devices()
    gpu_devices = []
    for device in device_list:
        if device.name.split(":")[1] == "GPU":
            text = device.physical_device_desc
            text = text.split("name: ", 1)[1]
            text = text.split(",", 1)[0]

            gpu_devices.append(text)
    return gpu_devices
get_gpu_devices()





# <a name="2-2"></a>
# ## Another GPU Availability Tests
# [ToC](#ToC)   
#
# Results from computer with GPU:
#
#     Num GPUs Available: 1
#
# Results from computer without GPU:  
#
#     Num GPUs Available: 0


print(f"Num GPUs Available: {len(list_physical_devices('GPU'))}")


# Results from computer with GPU:
#
#     ['/device:GPU:0']
#
# Results from computer without GPU:
#
#     []

def get_available_gpus():
    local_device_protos = list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
get_available_gpus()

# Results from computer with GPU:
#
#     [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
#
# Results from computer without GPU:
#
#     [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]

list_physical_devices("CPU")

# Results from computer with GPU:
#
#     [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
#
# Results from computer without GPU:
#     
#     []

list_physical_devices("GPU")

device_list = list_local_devices()
pprint(device_list)


# <a name="2-3"></a>
# # Get GPU Names
# [ToC](#ToC)  

# +
def get_gpu_devices() -> str:
    """
    Returns name of the GPU devices available.
    """
    device_list = list_local_devices()
    gpu_devices = []
    for device in device_list:
        if device.device_type == "GPU":
            text = list_local_devices()[1].physical_device_desc
            text = text.split("name: ", 1)[1]
            text = text.split(",", 1)[0]
            
            gpu_devices.append(text)
    return gpu_devices

get_gpu_devices()


# +
def get_gpu_devices() -> str:
    """
    Returns name of the GPU devices available.
    """
    device_list = list_local_devices()
    gpu_devices = []
    for device in device_list:
        if device.name.split(":")[1] == "GPU":
            text = device.physical_device_desc
            text = text.split("name: ", 1)[1]
            text = text.split(",", 1)[0]

            gpu_devices.append(text)
    return gpu_devices

get_gpu_devices()
# -

# <a name="3"></a>
# # Final Timestamp
# [ToC](#ToC)   

Logger().end_timer()
