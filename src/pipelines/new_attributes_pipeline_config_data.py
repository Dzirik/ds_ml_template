"""
Data structure for config.
"""
from typing import NamedTuple, List, Dict, Any


class NewAttribute(NamedTuple):
    """
    For storing parameters for creating new attributes.
    - create - Bool. if to create this attribute or not. The config can then contain all the possible attribute,
               and it can be just flagged which one to create.
    - name_of_transformation - str. Name of transformation to be used.
    - for_attrs - List[str]. Can be empty. But in case of application of one simple transformation on multiple
                             attributes. In case of not use, just empty list is put.
    - use_saved - bool. For example for oh different model has to be used for training and prediction (especially
                        for months for example). If not, fit_predict is used.
    - name_of_file - str. If to use saved, name of the transformer (without pkl) located in models path.
    - parameters - Dict[str, Any]. Dictionary of parameters.
    """
    create: bool
    name_of_transformation: str
    for_attrs: List[str]
    use_saved: bool
    name_of_file: str
    parameters: Dict[str, Any]


class NewAttributesPipelineConfigData(NamedTuple):
    """
    For storing parameters for pipeline.
    """
    name: str
    new_attributes: List[NewAttribute]
