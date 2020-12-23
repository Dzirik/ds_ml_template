# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # PostgreSQL Playground

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

import psycopg2

from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import sessionmaker

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


from src.data.postgre_connector import DBConnectorFactory

# Constants (please use all letter upper), ...

from src.global_constants import *  # Remember to import only the constants in use

# # ANALYSIS ---------------------------------------------------------

# ## Local Database

# ### Database Installation
#
# First of all, install Posgre SQL locally and create a database *test* there. Use following links for that:
# * [Install postgreSQL](https://www.postgresqltutorial.com/install-postgresql/)
# * [Create database test](https://www.postgresqltutorial.com/postgresql-python/connect/). Please use the SQL Shell windows console, log to data base and insert command `CREATE DATABASE test;`

# ### Connection to Database
#
# A very simple connection class of facade pattern is used for that. The configureation is read from config file. **Please check your credentials there.**


db_connection = DBConnectorFactory().create().get_connection()

# ### Operations

# #### Create Table

base = declarative_base()
class Film(base):
    __tablename__ = "films"
    
    id = Column(Integer, primary_key=True)
    title = Column(String)
    director = Column(String)
    year = Column(String)


Session = sessionmaker(db_connection)
session = Session()
base.metadata.create_all(db_connection)

# ####  Add Records

doctor_strange = Film(title="Doctor Strange", director="Scott Derrickson", year="2016")  
session.add(doctor_strange)  
session.commit()

batman = Film(title="Batman Begins", director="Chris Nolan", year="2005")  
session.add(batman)  
session.commit()


# #### Read

def print_films():
    films = session.query(Film)  
    for film in films:  
        print(f"Title: {film.title}, Director: {film.director}, Year: {film.year}")


print_films()

# #### Update

# Update
doctor_strange.title = "Some2016Film"  
session.commit()

print_films()

# #### Delete

# Delete
session.delete(doctor_strange)  
session.commit()  

print_films()

# ### Close Connection

db_connection.dispose()

# ## Final Timestamp

Logger().end_timer()
