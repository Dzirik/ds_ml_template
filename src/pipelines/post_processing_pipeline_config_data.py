"""
Data structure for config.
"""

from typing import NamedTuple, Dict, Any, List


class PostProcessingOperation(NamedTuple):
    """
    For storing parameters for data frame post-processing.
    - create - Bool; if to create this attribute or not. The config can then contain all the possible attribute,
               and it can be just flagged which one to create.
    - name_of_attribute - str. Name of operation to be done.
    - parameters - Dict[str, Any]. Dictionary of parameters.
    """
    create: bool
    name_of_operation: str
    parameters: Dict[str, Any]


class PostProcessingPipelineConfigData(NamedTuple):
    """
    For storing parameters for pipeline.
    """
    name: str
    post_processing_operations: List[PostProcessingOperation]
