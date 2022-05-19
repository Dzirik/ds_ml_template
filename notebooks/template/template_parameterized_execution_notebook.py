# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# # Template Parameterized Execution Notebook

# There is a usage of the [papermill](https://github.com/nteract/papermill) library in the core of this functionality. Please see the page and read about the usage, it is well written and documented. Hints:
# * First, turn on the tabs option in View/Cells Toolbar/Tags. <img src="..\..\assets\par_ntb_tag.png">.
# * Second, add a *parameters* tag to the cell where the selected variables to be parameterized are. If not specified/tagged, the parameters will be added as a separate cell at the top of the notebook. <img src="..\..\assets\par_ntb_tag_add.png">  
# * The added tab can be seen at the top of the cell. <img src="..\..\assets\par_ntb_tag_added.png"> 
# * Run the script; tested from PyCharm and it worked. From console is a problem with the path.
#
# > **INSTALATION NOTE:** Problems with pywin32 library was encountered with Anaconda 3.8. Downgrade to pywin32==225 helped.

# ## Imports

import matplotlib.pyplot as plt
# %matplotlib inline

# ## Parameters

# + {"tags": ["parameters"]}
n = 20
a = 1
b = 0
title = "Title"
# -

# ## Data

n = int(n) # when using config, there is a trouble with conversion
X = list(range(1, n+1))
Y = [a*x + b for x in X]

# ## Plotting

plt.plot(X, Y, "y.")
plt.title(title)
