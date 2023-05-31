"""
Transformer
"""

from typing import Any

from numpy import ndarray, dtype, double, sign, array, log, exp

from src.transformations.base_transformer import BaseTransformer, TransformerDescription
from src.transformations.log_transformer import LogTransformer


class LogNegativeTransformer(BaseTransformer):  # type:ignore
    """
    Computes sign(x) * log(x+1) transformation. E.g. "logarithm" of both positive and negative numbers.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(
            input_type=[ndarray[Any, dtype[double]]], input_elements_type=[float],
            output_type=[ndarray[Any, dtype[double]]], output_elements_type=[float]
        )
        BaseTransformer.__init__(
            self, class_name="LogNegativeTransformer", transformer_description=transformer_description
        )

        self._log_transformer = LogTransformer()

    def _transform(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Performs the sign(x) * log(x + 1) transformation for an array.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :return: Union[DataFrame, Series].
        """
        out = self._log_transformer.fit_predict(abs(data), self._params["base"])
        return array(out * sign(data))

    # pylint: disable=arguments-differ
    def fit(self, data: ndarray[Any, dtype[double]], base: float) -> None:
        """
        Fits.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :param base: float. Base of the logarithm.
        """
        self._params = {
            "base": base
        }

    def fit_predict(self, data: ndarray[Any, dtype[double]], base: float) -> ndarray[Any, dtype[double]]:
        """
        Fits and predicts.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :param base: float. Base of the logarithm.
        :return: ndarray[Any, dtype[double]].
        """
        self.fit(data, base)
        return self.predict(data)

    def predict(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :return: ndarray[Any, dtype[double]].
        """
        return self._transform(data)

    def inverse(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Does the inverse transformation.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :return: ndarray[Any, dtype[double]].
        """
        output: ndarray[Any, dtype[double]] = sign(data) * (exp(abs(data) * log(self._params["base"])) - 1.)
        return output

    # pylint: enable=arguments-differ
