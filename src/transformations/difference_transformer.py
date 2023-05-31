"""
Transformer
"""

from typing import Any

from numpy import ndarray, dtype, double, concatenate, array
from pandas import Series

from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class DifferenceTransformer(BaseTransformer):  # type:ignore
    """
    Does the k-th difference of array and returns:
    - zeros instead of NaNs,
    - differences then.
    So the length is the same as original data.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(
            input_type=[ndarray[Any, dtype[double]]], input_elements_type=[float],
            output_type=[ndarray[Any, dtype[double]]], output_elements_type=[float]
        )
        BaseTransformer.__init__(
            self, class_name="DifferenceTransformer", transformer_description=transformer_description
        )

    # pylint: disable=arguments-differ
    def fit(self, data: ndarray[Any, dtype[double]], periods: int = 1) -> None:
        """
        Fits.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
        :param periods: int. Order of the difference.
        """
        self._params = {
            "periods": periods
        }

    def fit_predict(self, data: ndarray[Any, dtype[double]], periods: int = 1) -> ndarray[Any, dtype[double]]:
        """
        Fits and predicts.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
        :param periods: int. Order of the difference.
        :return: ndarray[Any, dtype[double]].
        """
        self.fit(data, periods)
        return self.predict(data)

    def predict(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
        :return: ndarray[Any, dtype[double]].
        """
        ts = Series(data.reshape([data.shape[0], ]))
        ts_out = ts.diff(periods=self._params["periods"])
        ts_out = ts_out[ts_out.notnull()]
        return concatenate([array([0.] * self._params["periods"]), ts_out.to_numpy()]).reshape([data.shape[0], 1])

    def inverse(self) -> Any:
        """
        Does the inverse transformation.
        """
        return None

    # pylint: enable=arguments-differ
