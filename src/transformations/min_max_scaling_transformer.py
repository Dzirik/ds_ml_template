"""
Transformer

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
"""

from typing import Any, Tuple, Dict

from numpy import ndarray, dtype, std, double, array, repeat
from sklearn.preprocessing import MinMaxScaler

from src.exceptions.data_exception import IncorrectDataStructure
from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class MinMaxScalingTransformer(BaseTransformer):  # type:ignore
    """
    Scales an array of values within a given range (default = [0, 1]). Both possibilities to do it for all
    columns separately or together is possible.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0., 1.)) -> None:
        transformer_description = TransformerDescription(
            input_type=[ndarray[Any, dtype[double]]], input_elements_type=[float],
            output_type=[ndarray[Any, dtype[double]]], output_elements_type=[float]
        )
        BaseTransformer.__init__(self, class_name="MinMaxScalingTransformer",
                                 transformer_description=transformer_description)
        self._feature_range = feature_range
        self._scaler: MinMaxScaler

    # pylint: disable=arguments-differ
    def fit(self, data: ndarray[Any, dtype[double]], single_columns: bool = True) -> None:
        """
        Fits.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :param single_columns: bool. If to do the transformation separately for single columns or not.
        """
        for element in std(data, axis=0):
            if element == 0:
                raise IncorrectDataStructure

        if single_columns:
            self._scaler = MinMaxScaler(feature_range=self._feature_range)
            self._scaler.fit(X=data)
            self._params = {
                "main_params": self._scaler.get_params(deep=True),
                "min_": self._scaler.min_,
                "scale_": self._scaler.scale_,
                "data_min_": self._scaler.data_min_,
                "data_max_": self._scaler.data_max_
            }
        else:
            p = data.shape[1]
            global_min_max_data: ndarray[Any, dtype[double]] = array([[data.min()], [data.max()]])
            scaler = MinMaxScaler(feature_range=self._feature_range)
            scaler.fit(global_min_max_data)
            self._params = {
                "main_params": scaler.get_params(deep=True),
                "min_": repeat(scaler.min_, p),
                "scale_": repeat(scaler.scale_, p),
                "data_min_": repeat(scaler.data_min_, p),
                "data_max_": repeat(scaler.data_max_, p)
            }
            self.restore_from_params(self._params)

    def fit_predict(self, data: ndarray[Any, dtype[double]], single_columns: bool = True) -> \
            ndarray[Any, dtype[double]]:
        """
        Fits and predicts.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :return: Any.
        """
        self.fit(data, single_columns)
        return self.predict(data)

    def predict(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :return: ndarray[Any, dtype[double]].
        """
        out: ndarray[Any, dtype[double]] = self._scaler.transform(data)
        return out

    def inverse(self, data: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Does the inverse transformation.
        :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
        :return: ndarray[Any, dtype[double]]
        """
        out: ndarray[Any, dtype[double]] = self._scaler.inverse_transform(data)
        return out

    def restore_from_params(self, params: Dict[str, Any]) -> None:
        """
        Sets the params for transformer.
        :param params: Dict[str, Any]. See the fit method for more details.
        """
        self._params = params

        self._scaler = MinMaxScaler(self._params["main_params"])

        self._scaler.min_ = array(self._params["min_"])
        self._scaler.scale_ = array(self._params["scale_"])
        self._scaler.data_min_ = array(self._params["data_min_"])
        self._scaler.data_max_ = array(self._params["data_max_"])

    # pylint: enable=arguments-differ
