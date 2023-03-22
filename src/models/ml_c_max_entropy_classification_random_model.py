"""
ML Model
"""

from random import randrange, seed
from typing import Dict, Any

from numpy import ndarray, dtype, double, eye, array, float32

from src.exceptions.development_exception import NoProperOptionInIf
from src.models.base_ml_model import BaseMLModel, MLModelDescription


class MaxEntropyClassificationRandomModel(BaseMLModel):  # type:ignore
    """
    Linear regression model from StatsModel library.
    """

    def __init__(self) -> None:
        ml_model_description = MLModelDescription(
            type_of_model="custom",
            type_of_task="classification"
        )
        BaseMLModel.__init__(self, class_name="MaxEntropyClassificationRandomModel", model_class=None,
                             ml_model_description=ml_model_description)

        self._seed_number = 876
        self._n_of_classes: int  # Assumes number of classes to be Y.max() + 1, numbered from 0.

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
        :param Y: ndarray[Any, dtype[double]]. Dependent variable matrix, n x 1 numpy array. Assumes number of classes
                  to be Y.max() + 1, numbered from 0.
        :param model_params: Dict[str, Any]. Model parameters as a dictionary.
        """
        self._model_params = model_params

        if Y.shape[1] == 1 and not self._model_params["oh"]:
            self._n_of_classes = Y.max() + 1
        elif Y.shape[1] > 1 and self._model_params["oh"]:
            self._n_of_classes = Y.shape[1]
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
            predictions = [eye(1, self._n_of_classes, randrange(0, self._n_of_classes)).reshape((self._n_of_classes,))
                          for _ in range(X.shape[0])]
            return array(predictions, dtype=float32).reshape((X.shape[0], self._n_of_classes))
        predictions = [array(randrange(0, self._n_of_classes)) for _ in range(X.shape[0])]
        return array(predictions, dtype=float32).reshape((X.shape[0], 1))
