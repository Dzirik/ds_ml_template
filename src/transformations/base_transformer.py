"""
Parent/base class for transformers as parent class.

Transformers are classes for data transformation and manipulation. They are not intended to be machine learning models.

IMPORTANT NOTE: Data consistency is not checked here, the data input can be any and needs to be specified in child
classes. Solution which would handle it in general in python environment wold be too time consuming.
"""

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, List


class TransformerDataDescription(NamedTuple):
    """
    Tuple for storing input and output data type.
    """
    input_type: List[Any]
    input_elements_type: List[Any]
    output_type: List[Any]
    output_elements_type: List[Any]


class TransformerClassInfo(NamedTuple):
    """
    Tuple for storing info about Transformer.
    """
    class_type: Any
    class_name: Any
    data: TransformerDataDescription


class BaseTransformer(ABC):
    """
    Parent/base class for all transformers to ensure the same interface.
    """

    def __init__(self, class_name: str, data_description: TransformerDataDescription) -> None:
        self._info = TransformerClassInfo(
            class_type="Transformer",
            class_name=class_name,
            data=data_description
        )

    def get_info(self) -> TransformerClassInfo:
        """
        Returns class information named tuple.
        :return: TransformerClassInfo. Class information.
        """
        return self._info

    @abstractmethod
    def fit(self, data: Any) -> None:
        """
        Fits a transformation based on the data - analogy to sklearn style.
        :param data: Any. Not specified here.
        """

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """
        Does the transformation based on the fit - analogy to sklearn style.
        """

    @abstractmethod
    def fit_predict(self, data: Any) -> Any:
        """
        Does both fit and predict together - analogy to sklearn style.
        """
