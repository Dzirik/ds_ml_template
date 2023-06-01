"""
Data structure for config.
"""
from typing import NamedTuple, List, Optional, Dict


class RowBlocksGrouping(NamedTuple):
    """
    For storing grouping parameters for block grouping.
    - create - Bool; if to create this attribute or not. The config can then contain all the possible attribute,
               and it can be just flagged which one to create.
    - attrs - List of parameters to be grouped. Datetime has to be included as first one.
    - fun - Grouping function from transformer.
    - rename - Dictionary for renaming the column. If None, nothing is renamed.
    """
    create: bool
    attrs: List[str]
    fun: str
    rename: Optional[Dict[str, str]]


class RowBlocksGroupingPipelineConfigData(NamedTuple):
    """
    For storing parameters for pipeline.
    """
    name: str
    grouping_window_len: str  # see transformer for more details
    row_blocks_grouping: List[RowBlocksGrouping]
