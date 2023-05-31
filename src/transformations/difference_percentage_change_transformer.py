"""
Transformer
"""

from typing import Any

from numpy import ndarray, dtype, double, concatenate, zeros, array

from src.transformations.base_transformer import BaseTransformer, TransformerDescription
from src.transformations.difference_transformer import DifferenceTransformer


class DifferencePercentageChangeTransformer(BaseTransformer):  # type:ignore
    """
    Computes:
    - difference (as in difference transformer),
    - percentage change with respect to the older value,
    - adds zeros to the beginning,
    - optionally an epsilon can be added to prevent by zero division.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(
            input_type=[ndarray[Any, dtype[double]]], input_elements_type=[float],
            output_type=[ndarray[Any, dtype[double]]], output_elements_type=[float]
        )
        BaseTransformer.__init__(
            self, class_name="DifferencePercentageChangeTransformer", transformer_description=transformer_description
        )

    # pylint: disable=arguments-differ
    def fit(self, data: ndarray[Any, dtype[double]], periods: int = 1, eps: float = 0.) -> None:
        """
        Fits.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
        :param periods: int. Order of the difference.
        :param eps: float. What to add to prevent division by zero.
        """
        self._params = {
            "periods": periods,
            "eps": eps
        }

    def fit_predict(self, data: ndarray[Any, dtype[double]], periods: int = 1, eps: float = 0.) -> \
            ndarray[Any, dtype[double]]:
        """
        Fits and predicts.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
        :param periods: int. Order of the difference.
        :param eps: float. What to add to prevent division by zero.
        :return: ndarray[Any, dtype[double]].
        """
        self.fit(data, periods, eps)
        return self.predict(data)

    def predict(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
        :return: ndarray[Any, dtype[double]].
        """
        # conversions because of reading data from config file
        periods = int(self._params["periods"])
        eps = float(self._params["eps"])

        difference_transformer = DifferenceTransformer()
        differences = difference_transformer.fit_predict(data, periods)
        if abs(eps) > 0.:
            difference_changes = differences[periods:, ] / (data[:-periods, ] + eps)
            correction_array: ndarray[Any, dtype[double]] = array(
                [int(not (float(i) == 0. and abs(float(j)) > 0.)) for i, j in zip(list(data[:-1]), list(data[1:]))])
            difference_changes = difference_changes * correction_array.reshape((difference_changes.shape[0], 1))
        else:
            difference_changes = differences[periods:, ] / data[:-periods, ]
        missing_zeros = zeros((periods)).reshape((periods, 1))

        return concatenate((missing_zeros, difference_changes), axis=0)

    def inverse(self) -> Any:
        """
        Does the inverse transformation.
        """
        return None

    # pylint: enable=arguments-differ
