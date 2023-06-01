"""
Tester

No inverse here.
No parameter restoration.
"""

from typing import Tuple, Any, Dict

import pytest
from numpy import array, ndarray, dtype, double, testing, quantile
from numpy.random import normal, seed

from src.constants.global_constants import FP, P
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.quantile_transformer import QuantileTransformer

DEFAULT_LOWER, DEFAULT_HIGHER = QuantileTransformer().get_defaults()
SIMPLE_ARRAY_1: ndarray[Any, dtype[double]] = array([float(i) for i in range(11, 0, -1)]).reshape((11, 1))
SIMPLE_ARRAY_2: ndarray[Any, dtype[double]] = array([5., 1., 3., 2., 4.]).reshape((5, 1))
SIMPLE_ARRAY_2_DICT = {
    5.: 1.,
    1.: 0.,
    3.: 0.5,
    2.: 0.25,
    4.: 0.75
}


def _do_transformation(type_of_method: str, data: ndarray[Any, dtype[double]]) -> \
        Tuple[ndarray[Any, dtype[double]], QuantileTransformer]:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. Two dimensional numpy array of the size (n,1) -
                     DataFrame[[attr_name]].to_numpy().
    :return: Tuple[ndarray[Any, dtype[double]]], LogTransformer].
    """
    transformer = QuantileTransformer()
    if type_of_method == FP:
        transformed_data = transformer.fit_predict(data)
    elif type_of_method == P:
        transformer.fit(data)
        transformed_data = transformer.predict(data)
    else:
        raise NoProperOptionInIf

    return transformed_data, transformer


seed(98656)


@pytest.mark.parametrize("type_of_method, data",
                         [
                             (FP, SIMPLE_ARRAY_1), (P, SIMPLE_ARRAY_1),
                             (FP, normal(size=11).reshape((11, 1))), (P, normal(size=11).reshape((11, 1))),
                             (FP, normal(size=21).reshape((21, 1))), (P, normal(size=21).reshape((21, 1))),
                             (FP, normal(size=31).reshape((31, 1))), (P, normal(size=31).reshape((31, 1))),
                             (FP, normal(size=41).reshape((41, 1))), (P, normal(size=41).reshape((41, 1)))
                         ])
def test_calculation(type_of_method: str, data: ndarray[Any, dtype[double]]) -> None:
    """
    Tests that the logarithms are being correctly calculated.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. Two dimensional numpy array of the size (n,1) -
                     DataFrame[[attr_name]].to_numpy().
    """
    data_out, _ = _do_transformation(type_of_method, data)
    quantile_results: ndarray[Any, dtype[double]] = array(quantile(data, data_out, method="higher"))
    quantile_results = quantile_results.reshape((data.shape[0], 1))
    testing.assert_equal(data.round(4), quantile_results.round(4))


@pytest.mark.parametrize("type_of_method, data, data_predict, true_out",
                         [
                             (FP, SIMPLE_ARRAY_2, array([[1.4], [1.5], [1.6]]), array([[0.], [0.], [0.25]])),
                             (P, SIMPLE_ARRAY_2, array([[1.4], [1.5], [1.6]]), array([[0.], [0.], [0.25]])),
                             (FP, SIMPLE_ARRAY_2, array([[5.1]]), array([[DEFAULT_HIGHER]])),
                             (P, SIMPLE_ARRAY_2, array([[5.1]]), array([[DEFAULT_HIGHER]])),
                             (FP, SIMPLE_ARRAY_2, array([[0.], [5.], [10.]]), array([[-1.], [1.], [2.]])),
                             (P, SIMPLE_ARRAY_2, array([[0.], [5.], [10.]]), array([[-1.], [1.], [2.]]))
                         ])
def test_out_of_train_data_predictions(type_of_method: str, data: ndarray[Any, dtype[double]],
                                       data_predict: ndarray[Any, dtype[double]], \
                                       true_out: ndarray[Any, dtype[double]]) \
        -> None:
    """
    Tests the dictionary output.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. Two dimensional numpy array of the size (n,1) -
                     DataFrame[[attr_name]].to_numpy().
    :param data_predict: ndarray[Any, dtype[double]]. Data to be predicted.
    :param true_out: ndarray[Any, dtype[double]]. True output.
    """
    _, transformer = _do_transformation(type_of_method, data)
    out = transformer.predict(data_predict)
    testing.assert_equal(out.round(4), true_out.round(4))


@pytest.mark.parametrize("type_of_method, data, true_dictionary",
                         [
                             (FP, SIMPLE_ARRAY_2, SIMPLE_ARRAY_2_DICT)
                         ])
def test_dictionary_train(type_of_method: str, data: ndarray[Any, dtype[double]], true_dictionary: Dict[float, float]) \
        -> None:
    """
    Tests the dictionary output.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. Two dimensional numpy array of the size (n,1) -
                     DataFrame[[attr_name]].to_numpy().
    :param true_dictionary: Dict[float, float].
    """
    _, transformer = _do_transformation(type_of_method, data)
    dict_out_train, dict_out_prediction = transformer.get_dicts()
    assert dict_out_train == true_dictionary and not dict_out_prediction


@pytest.mark.parametrize("type_of_method, data, true_dictionary",
                         [
                             (P, SIMPLE_ARRAY_2, SIMPLE_ARRAY_2_DICT)
                         ])
def test_dictionary_prediction(type_of_method: str, data: ndarray[Any, dtype[double]],
                               true_dictionary: Dict[float, float]) \
        -> None:
    """
    Tests the dictionary output.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. Two dimensional numpy array of the size (n,1) -
                     DataFrame[[attr_name]].to_numpy().
    :param true_dictionary: Dict[float, float].
    """
    _, transformer = _do_transformation(type_of_method, data)
    dict_out_train, dict_out_prediction = transformer.get_dicts()
    assert dict_out_train == true_dictionary and dict_out_prediction == true_dictionary


def test_none_inverse() -> None:
    """
    Test none in inverse
    """
    transformer = QuantileTransformer()
    assert transformer.inverse() is None
