"""
Tester
"""
from typing import Any, Tuple

import pytest
from numpy import ndarray, dtype, double, array, testing

from src.constants.global_constants import FP, P
from src.exceptions.data_exception import IncorrectDataStructure
from src.exceptions.development_exception import NoProperOptionInIf
from src.models.anomaly_detection_rules_model import AnomalyDetectionRulesModel, AnomalyDetectionParams
from src.models.anomaly_detection_rules_model import DetectionRuleParams

PARAMS_1 = AnomalyDetectionParams(
    rules_definition=[
        DetectionRuleParams(
            name="FirstWecoRule",
            params={"side": "both", "alpha": 1}
        ),
        DetectionRuleParams(
            name="SecondWecoRule",
            params={"side": "both", "alpha": 1.}
        ),
        DetectionRuleParams(
            name="ThirdWecoRule",
            params={"side": "both", "alpha": 1.}
        ),
        DetectionRuleParams(
            name="FourthWecoRule",
            params={"side": "both", "alpha": 1.}
        )
    ]
)
DATA_FIT_1: ndarray[Any, dtype[double]] = array(range(1, 11)).reshape((10, 1))
DATA_PREDICT_1: ndarray[Any, dtype[double]] = array([list(range(8))])


def _do_transformation(type_of_method: str, data_fit: ndarray[Any, dtype[double]], data_predict: \
        ndarray[Any, dtype[double]], setting: AnomalyDetectionParams) -> \
        Tuple[ndarray[Any, dtype[double]], AnomalyDetectionRulesModel]:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    :param single_columns: bool. If to do the transformation separately for single columns or not.
    :param feature_range: Tuple[float, float]. Range for output.
    :return: Tuple[ndarray[Any, dtype[double]], MinMaxScalingTransformer].
    """
    model = AnomalyDetectionRulesModel()
    if type_of_method == FP:
        data_output = model.fit_predict(data_fit, data_predict, setting)
    elif type_of_method == P:
        model.fit(data_fit, setting)
        data_output = model.predict(data_predict)
    else:
        raise NoProperOptionInIf(model.get_class_info().class_name)
    return data_output, model


@pytest.mark.parametrize("type_of_method, data_fit, data_predict, setting, correct_out",
                         [
                             (FP, DATA_FIT_1, DATA_PREDICT_1, PARAMS_1, array([[0, 1, 1, 0]])),
                             (P, DATA_FIT_1, DATA_PREDICT_1, PARAMS_1, array([[0, 1, 1, 0]]))
                         ])
def test_transformation(type_of_method: str, data_fit: ndarray[Any, dtype[double]], data_predict: \
        ndarray[Any, dtype[double]], setting: AnomalyDetectionParams, correct_out: ndarray[Any, dtype[double]]) \
        -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data_fit: ndarray[Any, dtype[double]]. Two dimensional array of the shape (n, 1).
    :param data_predict: ndarray[Any, dtype[double]]. Two dimensional array of the shape (m, p). Rows are
        sequences of residuals to be tested.
    :param setting: AnomalyDetectionParams. Parameters for the model.
    :param correct_out: ndarray[Any, dtype[double]]. Correct output.
    """
    out, _ = _do_transformation(type_of_method, data_fit, data_predict, setting)
    testing.assert_equal(out, correct_out)


def test_incorrect_data_structure() -> None:
    """
    Tests incorrect data structure.
    :return:
    """
    with pytest.raises(IncorrectDataStructure):
        _do_transformation(P, array([[1., 1.], [2., 2.]]), array([[1., 1.], [2., 2.]]), PARAMS_1)


@pytest.mark.parametrize("type_of_method",
                         [
                             (FP), (P)
                         ])
def test_inverse(type_of_method: str) -> None:
    """
    Tests inverse - it is not used here.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    """
    _, model = _do_transformation(type_of_method, DATA_FIT_1, DATA_PREDICT_1, PARAMS_1)
    assert model.inverse() is None


@pytest.mark.parametrize("type_of_method",
                         [
                             (FP), (P)
                         ])
def test_restoration(type_of_method: str) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    """
    out, model = _do_transformation(type_of_method, DATA_FIT_1, DATA_PREDICT_1, PARAMS_1)

    model_restored = AnomalyDetectionRulesModel()
    model_restored.restore_from_params(model.get_params())
    out_restored = model_restored.predict(DATA_PREDICT_1)

    testing.assert_equal(out, out_restored)


def test_get_names() -> None:
    """
    Tests getter for rule names.
    """
    _, model = _do_transformation(FP, DATA_FIT_1, DATA_PREDICT_1, PARAMS_1)
    assert model.get_rule_names() == ["FirstWecoRule", "SecondWecoRule", "ThirdWecoRule", "FourthWecoRule"]
