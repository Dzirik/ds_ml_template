"""
ML Model
"""

from collections import Counter, OrderedDict
from typing import Dict, Any

from numpy import ndarray, dtype, double, arange, array, float32, eye
from numpy.random import seed, choice

from src.exceptions.development_exception import NoProperOptionInIf
from src.models.base_ml_model import BaseMLModel, MLModelDescription


class DistributionClassificationRandomModel(BaseMLModel):  # type:ignore
    """
    Linear regression model from StatsModel library.
    """

    def __init__(self) -> None:
        ml_model_description = MLModelDescription(
            type_of_model="custom",
            type_of_task="classification"
        )
        BaseMLModel.__init__(self, class_name="Linear Regression", model_class=None,
                             ml_model_description=ml_model_description)

        self._seed_number = 87698
        self._probability_distribution: ndarray[Any, dtype[double]]

    def set_seed_value(self, seed_value: int) -> None:
        """
        Sets the seed value.
        :param seed_value: int.
        """
        self._seed_number = seed_value

    def fit(self, X: ndarray[Any, dtype[double]], Y: ndarray[Any, dtype[double]], model_params: Dict[str, Any]) -> None:
        """
        Fits the model based on the data.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :param Y: ndarray[Any, dtype[double]]. Dependent variable matrix, n x 1 numpy array.
        :param model_params: Dict[str, Any]. Model parameters as a dictionary.
        """
        self._model_params = model_params

        if Y.shape[1] == 1 and not self._model_params["oh"]:
            d = dict(Counter(Y.reshape((Y.shape[0],))))
            freq = OrderedDict(sorted(d.items()))
            values = list(freq.values())
            self._probability_distribution = array([v / sum(values) for v in values])
        elif Y.shape[1] > 1 and self._model_params["oh"]:
            self._probability_distribution = Y.sum(axis=0) / Y.sum()
        else:
            raise NoProperOptionInIf

    def predict(self, X: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts based on the fitted model.

        NOTE: Overwriting the base class method because of the intercept.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :return: ndarray[Any, dtype[double]]. Prediction.
        """
        seed(self._seed_number)
        if self._model_params["oh"]:
            predictions = [eye(1, self._probability_distribution.shape[0], choice(
                arange(0, self._probability_distribution.shape[0]), p=self._probability_distribution))
                           for _ in range(X.shape[0])]
            return array(predictions, dtype=float32).reshape((X.shape[0], self._probability_distribution.shape[0]))
        predictions = [choice(arange(0, self._probability_distribution.shape[0]), p=self._probability_distribution)
                       for _ in range(X.shape[0])]
        return array(predictions, dtype=float32).reshape((X.shape[0], 1))
