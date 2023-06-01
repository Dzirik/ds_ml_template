"""
Tester
"""

from typing import Tuple, Any

import pytest
from numpy import ndarray, dtype, double, array, concatenate, testing

from src.constants.global_constants import FP, P
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.difference_transformer import DifferenceTransformer
from src.visualisations.visualisation_functions import create_time_series


def create_data_for_testing() -> Tuple[ndarray[Any, dtype[double]], ndarray[Any, dtype[double]]]:
    """
    Creates numpy arrays for testing.
    :return: Tuple[ndarray[Any, dtype[double]], ndarray[Any, dtype[double]]].
    """
    ts = create_time_series()
    data_1: ndarray[Any, dtype[double]] = array(list(range(1, ts.shape[0] + 1)), dtype="float64")
    data_2: ndarray[Any, dtype[double]] = ts.to_numpy()
    return data_1.reshape([ts.shape[0], 1]), data_2.reshape([ts.shape[0], 1])


DATA_1, DATA_2 = create_data_for_testing()


def _do_transformation(type_of_method: str, data: ndarray[Any, dtype[double]], periods: int) -> \
        Tuple[ndarray[Any, dtype[double]], DifferenceTransformer]:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
    :param periods: int. Length of the shift.
    :return: Tuple[ndarray[Any, dtype[double]], LogTransformer].
    """
    transformer = DifferenceTransformer()
    if type_of_method == FP:
        transformed_data = transformer.fit_predict(data, periods)
    elif type_of_method == P:
        transformer.fit(data, periods)
        transformed_data = transformer.predict(data)
    else:
        raise NoProperOptionInIf

    return transformed_data, transformer


def create_differences_by_hand(data: ndarray[Any, dtype[double]], periods: int) -> ndarray[Any, dtype[double]]:
    """
    Creates differences by hand.
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
    :param periods: int. Length of the shift.
    :return: ndarray[Any, dtype[double]].
    """
    data = data.reshape([data.shape[0], ])
    out: ndarray[Any, dtype[double]] = concatenate([array([0.] * periods), data[periods:] - data[:-periods]])
    return out.reshape([data.shape[0], 1])


@pytest.mark.parametrize("type_of_method, data, periods",
                         [
                             (FP, DATA_1, 1),
                             (P, DATA_1, 1),
                             (FP, DATA_1, 2),
                             (P, DATA_1, 2),
                             (FP, DATA_1, 5),
                             (P, DATA_1, 5),
                             (FP, DATA_1, 10),
                             (P, DATA_1, 10),
                             (FP, DATA_2, 1),
                             (P, DATA_2, 1),
                             (FP, DATA_2, 2),
                             (P, DATA_2, 2),
                             (FP, DATA_2, 5),
                             (P, DATA_2, 5),
                             (FP, DATA_2, 10),
                             (P, DATA_2, 10),
                         ])
def test_calculation(type_of_method: str, data: ndarray[Any, dtype[double]], periods: int) -> None:
    """
    Tests that the logarithms are being correctly calculated.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
    :param periods: int. Length of the shift.
    """
    data_out, _ = _do_transformation(type_of_method, data, periods)
    data_out_by_hand = create_differences_by_hand(data, periods)
    testing.assert_equal(data_out.round(4), data_out_by_hand.round(4))


@pytest.mark.parametrize("type_of_method, data, periods",
                         [
                             (FP, DATA_1, 1),
                             (P, DATA_1, 1),
                             (FP, DATA_1, 2),
                             (P, DATA_1, 2),
                             (FP, DATA_1, 5),
                             (P, DATA_1, 5),
                             (FP, DATA_1, 10),
                             (P, DATA_1, 10),
                             (FP, DATA_2, 1),
                             (P, DATA_2, 1),
                             (FP, DATA_2, 2),
                             (P, DATA_2, 2),
                             (FP, DATA_2, 5),
                             (P, DATA_2, 5),
                             (FP, DATA_2, 10),
                             (P, DATA_2, 10),
                         ])
def test_shape(type_of_method: str, data: ndarray[Any, dtype[double]], periods: int) -> None:
    """
    Tests that the logarithms are being correctly calculated.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    ::param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
    :param periods: int. Length of the shift.
    """
    data_out, _ = _do_transformation(type_of_method, data, periods)
    assert data_out.shape[0] == data.shape[0]


@pytest.mark.parametrize("type_of_method, data, periods",
                         [
                             (FP, DATA_1, 2),
                             (P, DATA_1, 2),
                             (FP, DATA_1, 5),
                             (P, DATA_1, 5)
                         ])
def test_restoration(type_of_method: str, data: ndarray[Any, dtype[double]], periods: int) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows and ONE column.
    :param periods: int. Length of the shift.
    """
    output_data, transformer = _do_transformation(type_of_method, data, periods)

    transformer_restored = DifferenceTransformer()
    transformer_restored.restore_from_params(transformer.get_params())
    output_data_restored = transformer_restored.predict(data)

    testing.assert_equal(output_data.round(4), output_data_restored.round(4))


def test_none_inverse() -> None:
    """
    Test none in inverse
    """
    transformer = DifferenceTransformer()
    assert transformer.inverse() is None
