"""
ML Model
"""

from typing import Dict, Any

import statsmodels.api as sm
from numpy import ndarray, dtype, double, ones, concatenate
from sklearn.metrics import r2_score

from src.models.base_ml_model import BaseMLModel, MLModelDescription


class StatsLinRegModel(BaseMLModel):  # type:ignore
    """
    Linear regression model from StatsModel library.
    """

    def __init__(self) -> None:
        ml_model_description = MLModelDescription(
            type_of_model="statsmodels",
            type_of_task="regression"
        )
        BaseMLModel.__init__(self, class_name="ModelStatsLinReg", model_class=sm.OLS,
                             ml_model_description=ml_model_description)

    def _add_intercept(self, X: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Updates x_data with column of ones in order to calculate intercept in LR model.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :return: ndarray: Updated x_data.
        """
        if self._model_params["intercept"]:
            X = concatenate((ones((X.shape[0], 1)), X), axis=1)
            # return sm.tools.add_constant(X) # it didn't work in predict stage
        return X

    def fit(self, X: ndarray[Any, dtype[double]], Y: ndarray[Any, dtype[double]], model_params: Dict[str, Any]) -> None:
        """
        Fits the model based on the data.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :param Y: ndarray[Any, dtype[double]]. Dependent variable matrix, n x 1 numpy array.
        :param model_params: Dict[str, Any]. Model parameters as a dictionary.
        """
        self._model_params = model_params
        self._model = self._model_class(Y.reshape((Y.shape[0],)), self._add_intercept(X)).fit()

    def predict(self, X: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts based on the fitted model.

        NOTE: Overwriting the base class method because of the intercept.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :return: ndarray[Any, dtype[double]]. Prediction.
        """
        prediction: ndarray[Any, dtype[double]] = self._model.predict(self._add_intercept(X))
        return prediction.reshape((X.shape[0], 1))

    def get_r2_score(self, X: ndarray[Any, dtype[double]], Y: ndarray[Any, dtype[double]]) -> float:
        """
        Returns the R2 score for (X,y) pair based on the fitted model.

        NOTE: Overwriting the base class method because of the intercept.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :param Y: ndarray[Any, dtype[double]]. Dependent variable matrix, n x 1 numpy array.
        :return: float. R2 score.
        """
        n = X.shape[0]
        Y_hat = self.predict(X)
        score = float(r2_score(y_true=Y.reshape((n,)), y_pred=Y_hat.reshape((n,))))
        return score
