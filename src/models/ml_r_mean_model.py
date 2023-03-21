"""
ML Model
"""

from typing import Dict, Any

from numpy import ndarray, dtype, double, array, float32

from src.models.base_ml_model import BaseMLModel, MLModelDescription


class MeanModel(BaseMLModel):  # type:ignore
    """
    Linear regression model from StatsModel library.
    """

    def __init__(self) -> None:
        ml_model_description = MLModelDescription(
            type_of_model="custom",
            type_of_task="regression"
        )
        BaseMLModel.__init__(self, class_name="MeanModel", model_class=None,
                             ml_model_description=ml_model_description)
        self._mean: ndarray[Any, dtype[double]]

    def fit(self, X: ndarray[Any, dtype[double]], Y: ndarray[Any, dtype[double]], model_params: Dict[str, Any]) -> None:
        """
        Fits the model based on the data.
        NOTE: X is used for the sake of consistency of output.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :param Y: ndarray[Any, dtype[double]]. Dependent variable matrix, n x 1 numpy array.
        :param model_params: Dict[str, Any]. Model parameters as a dictionary.
        """
        self._model_params = model_params
        self._mean = Y.mean(axis=0)

    def predict(self, X: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts based on the fitted model.

        NOTE: Overwriting the base class method because of the intercept.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :return: ndarray[Any, dtype[double]]. Prediction.
        """
        return array([self._mean] * X.shape[0], dtype=float32).reshape((X.shape[0], 1))
