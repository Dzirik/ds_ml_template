"""
Data structure for config.
"""
from typing import NamedTuple, List

from src.transformations.transformations_executioner_transformer import TransformerConfiguration


class TransformationExecutionerConfiguration(NamedTuple):
    """
    Data structure for holding configuration.
    """
    create: bool
    attrs: List[str]
    configurations: List[TransformerConfiguration]


class TransformationsExecutionerPipelineConfigData(NamedTuple):
    """
    Data structure for holding set up.
    """
    name: str
    trans_executioners_confs: List[TransformationExecutionerConfiguration]
