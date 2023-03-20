"""
Parent class for machine learning models of different origin:
- sk-learn,
- stats models,
- DNN.
"""

from abc import abstractmethod
from typing import Any, Dict

from numpy import ndarray, dtype, double
from sklearn.metrics import r2_score

from src.utils.meta_class import MetaClass, ML_MODEL_TYPE_NAME, MLModelDescription


class BaseMLModel(MetaClass):  # type:ignore
    """
    Parent/base class for all transformers to ensure the same interface.
    """

    def __init__(self, class_name: str, model_class: Any, ml_model_description: MLModelDescription) -> None:
        MetaClass.__init__(self, class_type=ML_MODEL_TYPE_NAME, class_name=class_name)
        self.set_ml_model_description(ml_model_description=ml_model_description)

        self._model_class = model_class  # sklearn or stats model or any other ML model class
        self._model_params: Dict[str, Any] = {}
        self._model: Any

    @abstractmethod
    def fit(self, X: ndarray[Any, dtype[double]], Y: ndarray[Any, dtype[double]], model_params: Dict[str, Any]) -> None:
        """
        Fits the model based on the data.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :param Y: ndarray[Any, dtype[double]]. Dependent variable matrix, n x 1 numpy array.
        :param model_params: Dict[str, Any]. Model parameters as a dictionary.
        """

    def predict(self, X: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts based on the fitted model.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :return: ndarray[Any, dtype[double]]. Prediction.
        """
        return self._model.predict(X)  # type:ignore

    def fit_predict(self, X: ndarray[Any, dtype[double]], Y: ndarray[Any, dtype[double]], \
                    model_params: Dict[str, Any]) -> ndarray[Any, dtype[double]]:
        """
        Fits the model for (X,y) data and does and returns the prediction for X data.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :param Y: ndarray[Any, dtype[double]]. Dependent variable matrix, n x 1 numpy array.
        :param model_params: Dict[str, Any]. Model parameters as a dictionary.
        :return: ndarray[Any, dtype[double]]. Prediction.
        """
        self.fit(X, Y, model_params)
        return self.predict(X)

    def get_r2_score(self, X: ndarray[Any, dtype[double]], Y: ndarray[Any, dtype[double]]) -> float:
        """
        Returns the R2 score for (X,y) pair based on the fitted model.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :param Y: ndarray[Any, dtype[double]]. Dependent variable matrix, n x 1 numpy array.
        :return: float. R2 score.
        """
        Y_hat = self.predict(X)
        score = float(r2_score(y_true=Y, y_pred=Y_hat))
        return score

    def get_residuals(self, X: ndarray[Any, dtype[double]], Y: ndarray[Any, dtype[double]]) -> \
            ndarray[Any, dtype[double]]:
        """
        Creates and returns residuals for the X and Y pair.
        :param X: ndarray[Any, dtype[double]]. Independent variable matrix, n x p numpy array.
        :param Y: ndarray[Any, dtype[double]]. Dependent variable matrix, n x 1 numpy array.
        :return: ndarray[Any, dtype[double]]. n x 1 numpy array.
        """
        Y_hat = self.predict(X)
        residuals = Y - Y_hat
        return residuals.reshape((X.shape[0], 1))

    def get_model(self) -> Any:
        """
        Gets the fitted model.
        :return: Any. Model
        """
        return self._model
