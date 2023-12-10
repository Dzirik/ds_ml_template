"""
Module for storing attribute names.

The goal is to have all data from different sources unified as much as possible.

All the attribute names and types are stored in A variable.
"""
from typing import NamedTuple, Any


class Attribute(NamedTuple):
    """
    For storing attribute and its data type.
    """
    name: str
    type: Any


ATTRS_HOURS = [f"HOUR_{i}.0" for i in range(24)]
ATTRS_DAYS_OF_WEEK = [f"DAY_OF_WEEK_{i}.0" for i in range(7)]
ATTRS_MONTHS = [f"MONTH_{i}.0" for i in range(1, 13, 1)]

PREPOSITION_DIFF = "DIFF"
PREPOSITION_DIFF_PERC = "DIFF_PERC"

ATTR_DATETIME = Attribute("Datetime".upper(), "datetime64[ns]")
ATTR_OPEN = Attribute("Open".upper(), "float32")
ATTR_HIGH = Attribute("High".upper(), "float32")
ATTR_LOW = Attribute("Low".upper(), "float32")
ATTR_CLOSE = Attribute("Close".upper(), "float32")


class Attributes(NamedTuple):
    """
    For storing all attributes for cry project.
    """
    # 01 attributes for yfinance and kraken data
    date_time: Attribute
    open: Attribute
    high: Attribute
    low: Attribute
    close: Attribute


A = Attributes(
    ATTR_DATETIME, ATTR_OPEN, ATTR_HIGH, ATTR_LOW, ATTR_CLOSE
)
