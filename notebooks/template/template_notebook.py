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

# # Template for Notebook

# *Please put you comments here.*
#
# *This notebook is equiped with:*   
# - *Jupytext.*   
# - *Time measurement.*  

# # GENERAL SETTINGS --------------------------------------

# ### General Libraries and Paths

import os
import sys

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload

# %matplotlib notebook
# # %matplotlib inline
# -

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))  # one up
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../..")))  # two up
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../..")))  # three up - for data sets

# ### Libraries Needed for the Notebook

from src.utils.notebook_support_functions import create_button, get_notebook_name
from src.utils.logger import Logger

NOTEBOOK_NAME = get_notebook_name()

# ### Notebook Apperance Settings

# - A button for hiding/showing the code.
# - Notebook data frame setting.
# - Initial timestamp.

create_button()

# +
pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

Logger().start_timer(f"NOTEBOOK; Notebook name: {NOTEBOOK_NAME}")
# -

# ### Personal Libraries

# Personal libraries, classes, functions from repo itself


# +
# Space for importing my own code.
# -

# Constants (please use all letter upper), ...

from src.constants.global_constants import *  # Remember to import only the constants in use

# # ANALYSIS ---------------------------------------------------------

# ## Chapter


Logger().set_meantime("Chapter one")

# ### Sub-Chapter


# ## Final Timestamp

Logger().end_timer()
