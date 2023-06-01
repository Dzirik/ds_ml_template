"""
Transformer
"""

from typing import Any

from numpy import ndarray, dtype, double

from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class ShiftByValueTransformer(BaseTransformer):  # type:ignore
    """
    Shifts all the values in the array by one value.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(
            input_type=[ndarray[Any, dtype[double]]], input_elements_type=[float],
            output_type=[ndarray[Any, dtype[double]]], output_elements_type=[float]
        )
        BaseTransformer.__init__(self, class_name="ShiftByValueTransformer",
                                 transformer_description=transformer_description)

    # pylint: disable=arguments-differ
    def fit(self, data: ndarray[Any, dtype[double]], shift_value: float) -> None:
        """
        Fits.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :param shift_value: float. Shift value.
        """
        self._params = {
            "shift_value": shift_value
        }

    def fit_predict(self, data: ndarray[Any, dtype[double]], shift_value: float) -> ndarray[Any, dtype[double]]:
        """
        Fits and predicts.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :param shift_value: float. Shift value.
        :return: ndarray[Any, dtype[double]].
        """
        self.fit(data, shift_value)
        return self.predict(data)

    def predict(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :return: ndarray[Any, dtype[double]].
        """
        return data + self._params["shift_value"]

    def inverse(self, data: Any) -> Any:
        """
        Does the inverse transformation.
        """
        return data - self._params["shift_value"]

    # pylint: enable=arguments-differ
