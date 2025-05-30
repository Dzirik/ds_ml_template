"""
Data structure for config.
"""
from typing import NamedTuple, List, Dict, Any


class ColumnsGrouping(NamedTuple):
    """
    For storing grouping parameters for column grouping.
    - create - Bool; if to create this attribute or not. The config can then contain all the possible attribute,
               and it can be just flagged which one to create.
    - attrs - List of parameters to be grouped. Datetime has to be included as first one.
    - grouping_fun - Grouping function from transformer.
    - params - detailed parameters for respective grouping.
    - new_attr_name - Name of the new column created.
    """
    create: bool
    attrs: List[str]
    grouping_fun: str
    params: Dict[str, Any]
    new_attr_name: str


class ColumnsGroupingPipelineConfigData(NamedTuple):
    """
    For storing parameters for pipeline.
    """
    name: str
    columns_grouping: List[ColumnsGrouping]
