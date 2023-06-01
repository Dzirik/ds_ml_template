"""
Tester
"""

from typing import Tuple, Any

import pytest
from numpy import ndarray, dtype, double, array, testing

from src.constants.global_constants import FP, P
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.difference_percentage_change_transformer import DifferencePercentageChangeTransformer

N = 30
EPS = 10**(-10)


def create_correct_data_for_testing(periods: int) -> ndarray[Any, dtype[double]]:
    """
    Creates correct output for difference for periods for 1-30 original data.
    :param periods: int. Periods for difference transformer.
    """
    return array([0.] * periods + [periods / i for i in range(1, N - periods + 1)]).reshape([N, 1])


DATA: ndarray[Any, dtype[double]] = array(list(range(1, N + 1)), dtype="float64").reshape((N, 1))
DATA_1_CORRECT = create_correct_data_for_testing(periods=1)
DATA_2_CORRECT = create_correct_data_for_testing(periods=2)
DATA_3_CORRECT = create_correct_data_for_testing(periods=3)
DATA_4_CORRECT = create_correct_data_for_testing(periods=4)
DATA_ZEROS: ndarray[Any, dtype[double]] = array([0., 0., 0., 1., 2., 3]).reshape((6, 1))
DATA_ZEROS_CORRECT: ndarray[Any, dtype[double]] = array([0., 0., 0., 0., 1., 0.5]).reshape((6, 1))


def _do_transformation(type_of_method: str, data: ndarray[Any, dtype[double]], periods: int, eps: float) -> \
        Tuple[ndarray[Any, dtype[double]], DifferencePercentageChangeTransformer]:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
    :param periods: int. Length of the shift.
    :param eps: float. What to add to prevent division by zero.
    :return: Tuple[ndarray[Any, dtype[double]], LogTransformer].
    """
    transformer = DifferencePercentageChangeTransformer()
    if type_of_method == FP:
        transformed_data = transformer.fit_predict(data, periods, eps)
    elif type_of_method == P:
        transformer.fit(data, periods, eps)
        transformed_data = transformer.predict(data)
    else:
        raise NoProperOptionInIf

    return transformed_data, transformer


@pytest.mark.parametrize("type_of_method, data, periods, data_correct, eps",
                         [
                             (FP, DATA, 1, DATA_1_CORRECT, 0.),
                             (P, DATA, 1, DATA_1_CORRECT, 0.),
                             (FP, DATA, 2, DATA_2_CORRECT, 0.),
                             (P, DATA, 2, DATA_2_CORRECT, 0.),
                             (FP, DATA, 3, DATA_3_CORRECT, 0.),
                             (P, DATA, 3, DATA_3_CORRECT, 0.),
                             (FP, DATA, 4, DATA_4_CORRECT, 0.),
                             (P, DATA, 4, DATA_4_CORRECT, 0.),
                             (FP, DATA_ZEROS, 1, DATA_ZEROS_CORRECT, EPS),
                             (P, DATA_ZEROS, 1, DATA_ZEROS_CORRECT, EPS)
                         ])
def test_calculation(type_of_method: str, data: ndarray[Any, dtype[double]], periods: int, data_correct: \
        ndarray[Any, dtype[double]], eps: float) -> None:
    """
    Tests that the logarithms are being correctly calculated.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
    :param periods: int. Length of the shift.
    :param eps: float. What to add to prevent division by zero.
    :param data_correct: ndarray[Any, dtype[double]]. Correct data array.
    """
    data_out, _ = _do_transformation(type_of_method, data, periods, eps)
    testing.assert_equal(data_out.round(4), data_correct.round(4))


@pytest.mark.parametrize("type_of_method, data, periods, eps",
                         [
                             (FP, DATA, 1, 0.),
                             (P, DATA, 1, 0.),
                             (FP, DATA, 2, 0.),
                             (P, DATA, 2, 0.),
                             (FP, DATA, 5, 0.),
                             (P, DATA, 5, 0.),
                             (FP, DATA, 10, 0.),
                             (P, DATA, 10, 0.),
                             (FP, DATA, 1, 0.),
                             (P, DATA, 1, 0.),
                             (FP, DATA, 2, 0.),
                             (P, DATA, 2, 0.),
                             (FP, DATA, 5, 0.),
                             (P, DATA, 5, 0.),
                             (FP, DATA, 10, 0.),
                             (P, DATA, 10, 0.),
                             (FP, DATA_ZEROS, 1, EPS),
                             (P, DATA_ZEROS, 1, EPS)
                         ])
def test_shape(type_of_method: str, data: ndarray[Any, dtype[double]], periods: int, eps: float) -> None:
    """
    Tests that the logarithms are being correctly calculated.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    ::param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
    :param periods: int. Length of the shift.
    """
    data_out, _ = _do_transformation(type_of_method, data, periods, eps)
    assert data_out.shape[0] == data.shape[0]


@pytest.mark.parametrize("type_of_method, data, periods, eps",
                         [
                             (FP, DATA, 2, 0.),
                             (P, DATA, 2, 0.),
                             (FP, DATA, 5, 0.),
                             (P, DATA, 5, 0.),
                             (FP, DATA_ZEROS, 1, EPS),
                             (P, DATA_ZEROS, 1, EPS)
                         ])
def test_restoration(type_of_method: str, data: ndarray[Any, dtype[double]], periods: int, eps: float) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
    :param periods: int. Length of the shift.
    :param eps: float. What to add to prevent division by zero.
    """
    output_data, transformer = _do_transformation(type_of_method, data, periods, eps)

    transformer_restored = DifferencePercentageChangeTransformer()
    transformer_restored.restore_from_params(transformer.get_params())
    output_data_restored = transformer_restored.predict(data)

    testing.assert_equal(output_data.round(4), output_data_restored.round(4))


def test_none_inverse() -> None:
    """
    Test none in inverse
    """
    transformer = DifferencePercentageChangeTransformer()
    assert transformer.inverse() is None
