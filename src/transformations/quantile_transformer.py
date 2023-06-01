"""
Transformer

NOTE: The implementation is not done effective with respect to memory and computation. It is an experimental
transformer, so it is done like that on purpose to test the concept.

Finding the closest value: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
"""

from typing import Any, Dict, Tuple

from numpy import ndarray, dtype, double, array

from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class QuantileTransformer(BaseTransformer):  # type:ignore
    """
    For each observation in 2d array of the size (n,1) - DataFrame[[attr_name]].to_numpy() computes its corresponding
    quantile value. The smallest observation gets 0, the largest one, and then they are equally distributed.

    The size (n,1) is chosen to be compatible with other transformers of this family.
    """
    _DEFAULT_HIGHER = 2.
    _DEFAULT_LOWER = -1.

    def __init__(self) -> None:
        transformer_description = TransformerDescription(
            input_type=[ndarray[Any, dtype[Any]]], input_elements_type=[double],
            output_type=[ndarray[Any, dtype[double]]], output_elements_type=[double]
        )
        BaseTransformer.__init__(
            self, class_name="QuantileTransformer", transformer_description=transformer_description
        )
        self._data: ndarray[Any, dtype[double]]
        self._quantiles: ndarray[Any, dtype[double]]
        self._train_dict: Dict[float, float] = {}
        self._prediction_dict: Dict[float, float] = {}

    def get_defaults(self) -> Tuple[float, float]:
        """
        Gets the default higher value.
        :return: Tuple[float, float]. Default lower and higher values.
        """
        return self._DEFAULT_LOWER, self._DEFAULT_HIGHER

    def get_dicts(self) -> Tuple[Dict[float, float], Dict[float, float]]:
        """
        Gets the dictionary original_value: quantile_value.
        :param data: ndarray[Any, dtype[double]]. Two dimensional numpy array of the size (n,1) -
                     DataFrame[[attr_name]].to_numpy().
        :return: Dict[float, float].
        """
        return self._train_dict, self._prediction_dict

    # pylint: disable=arguments-differ
    def fit(self, data: ndarray[Any, dtype[double]]) -> None:
        """
        Fits.
        :param data: ndarray[Any, dtype[double]]. Two dimensional numpy array of the size (n,1) -
                     DataFrame[[attr_name]].to_numpy().
        """
        self._data = data.reshape((data.shape[0],))

        n = len(self._data)

        sorted_data_indices = self._data.argsort().argsort()
        self._quantiles = array([i / (n - 1) for i in range(n)])
        self._quantiles = self._quantiles[sorted_data_indices]
        self._train_dict = dict(zip(self._data, self._quantiles))
        self._quantiles = self._quantiles.reshape((len(self._data), 1))

    def fit_predict(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Fits and predicts.
        :param data: ndarray[Any, dtype[double]]. Two dimensional numpy array of the size (n,1) -
                     DataFrame[[attr_name]].to_numpy().
        :return: ndarray[Any, dtype[double]].
        """
        self.fit(data)
        return self._quantiles

    def predict(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts.
        :param data: ndarray[Any, dtype[double]]. Two dimensional numpy array of the size (n,1) -
                     DataFrame[[attr_name]].to_numpy().
        :return: Any.
        """
        predictions = []
        for i in range(data.shape[0]):
            value = data[i, 0]
            if value in self._train_dict:
                predictions.append(self._train_dict[value])
            elif value > self._data.max():
                predictions.append(self._DEFAULT_HIGHER)
            elif value < self._data.min():
                predictions.append(self._DEFAULT_LOWER)
            else:
                idx = (abs(self._data - value)).argmin()
                predictions.append(self._quantiles[idx, 0])
        self._prediction_dict = dict(zip(data.reshape((data.shape[0],)), predictions))
        return array(predictions).reshape((data.shape))

    def inverse(self) -> None:
        """
        No inverse transformation here.
        """
        return None

    # pylint: enable=arguments-differ
