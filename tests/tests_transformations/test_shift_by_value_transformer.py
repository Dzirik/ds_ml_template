"""
Tester
"""
from typing import List, Any, Tuple

import pytest
from numpy import ndarray, dtype, double, array, testing

from src.constants.global_constants import FP, P
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.shift_by_value_transformer import ShiftByValueTransformer


def create_testing_array(bases: List[float]) -> ndarray[Any, dtype[double]]:
    """
    Creates a testing array.
    :param bases: List[float]. List of bases - attributes.
    :return:
    """
    n = 10
    data = []
    for i in range(n):
        data.append([b * i + b / 10. * 3. for b in bases])
    return array(data)


ARRAY_BASES: ndarray[Any, dtype[double]] = create_testing_array([10., 100., 1000.])
BASE_CORRECT_1: ndarray[Any, dtype[double]] = array([float(x) for x in range(0, 10)]).reshape((10, 1))
BASE_CORRECT_2: ndarray[Any, dtype[double]] = array([float(x) for x in range(1, 11)]).reshape((10, 1))


def _do_transformation(type_of_method: str, data: ndarray[Any, dtype[double]], shift_value: float) -> \
        Tuple[ndarray[Any, dtype[double]], ShiftByValueTransformer]:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    :param shift_value: float. Shift value.
    :return: Tuple[Union[DataFrame, Series], LogTransformer].
    """
    transformer = ShiftByValueTransformer()
    if type_of_method == FP:
        transformed_data = transformer.fit_predict(data, shift_value)
    elif type_of_method == P:
        transformer.fit(data, shift_value)
        transformed_data = transformer.predict(data)
    else:
        raise NoProperOptionInIf

    return transformed_data, transformer


@pytest.mark.parametrize("type_of_method, data, shift_value, correct_output",
                         [
                             (FP, ARRAY_BASES[:, 0].reshape((ARRAY_BASES.shape[0], 1)), - 3., BASE_CORRECT_1 * 10.),
                             (P, ARRAY_BASES[:, 0].reshape((ARRAY_BASES.shape[0], 1)), - 3., BASE_CORRECT_1 * 10.),
                             (FP, ARRAY_BASES[:, 1].reshape((ARRAY_BASES.shape[0], 1)), - 30., BASE_CORRECT_1 * 100.),
                             (P, ARRAY_BASES[:, 1].reshape((ARRAY_BASES.shape[0], 1)), - 30., BASE_CORRECT_1 * 100.),
                             (FP, ARRAY_BASES[:, 2].reshape((ARRAY_BASES.shape[0], 1)), - 300., BASE_CORRECT_1 * 1000.),
                             (P, ARRAY_BASES[:, 2].reshape((ARRAY_BASES.shape[0], 1)), - 300., BASE_CORRECT_1 * 1000.),
                             (FP, ARRAY_BASES[:, 0].reshape((ARRAY_BASES.shape[0], 1)), 7., BASE_CORRECT_2 * 10.),
                             (P, ARRAY_BASES[:, 0].reshape((ARRAY_BASES.shape[0], 1)), 7., BASE_CORRECT_2 * 10.),
                             (FP, ARRAY_BASES[:, 1].reshape((ARRAY_BASES.shape[0], 1)), 70., BASE_CORRECT_2 * 100.),
                             (P, ARRAY_BASES[:, 1].reshape((ARRAY_BASES.shape[0], 1)), 70., BASE_CORRECT_2 * 100.),
                             (FP, ARRAY_BASES[:, 2].reshape((ARRAY_BASES.shape[0], 1)), 700., BASE_CORRECT_2 * 1000.),
                             (P, ARRAY_BASES[:, 2].reshape((ARRAY_BASES.shape[0], 1)), 700., BASE_CORRECT_2 * 1000.),
                         ])
def test_calculation(type_of_method: str, data: ndarray[Any, dtype[double]], shift_value: float, \
                     correct_output: ndarray[Any, dtype[double]]) -> None:
    """
    Tests that the logarithms are being correctly calculated.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    :param shift_value: float. Shift value.
    :param correct_output: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    """
    data_out, _ = _do_transformation(type_of_method, data, shift_value)
    testing.assert_equal(data_out.round(4), correct_output.round(4))


@pytest.mark.parametrize("type_of_method, data, base",
                         [
                             (FP, ARRAY_BASES, 2.),
                             (P, ARRAY_BASES, 2.),
                             (FP, ARRAY_BASES, -100.),
                             (P, ARRAY_BASES, -100.)
                         ])
def test_inverse(type_of_method: str, data: ndarray[Any, dtype[double]], base: float) -> None:
    """
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    :param base: float. Base of the logarithm.
    """
    data_out, transformer = _do_transformation(type_of_method, data, base)
    data_out_inverse = transformer.inverse(data_out)
    testing.assert_equal(data.round(4), data_out_inverse.round(4))


@pytest.mark.parametrize("type_of_method, data_input, shift_value",
                         [
                             (FP, ARRAY_BASES, 5.),
                             (P, ARRAY_BASES, 5.),
                             (FP, ARRAY_BASES, - 100.),
                             (P, ARRAY_BASES, - 100.)
                         ])
def test_restoration(type_of_method: str, data_input: ndarray[Any, dtype[double]], shift_value: float) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data_input: ndarray[Any, dtype[double]].
    :param shift_value: float. Shift value.
    """
    output_data, transformer = _do_transformation(type_of_method, data_input, shift_value)

    transformer_restored = ShiftByValueTransformer()
    transformer_restored.restore_from_params(transformer.get_params())
    output_data_restored = transformer_restored.predict(data_input)

    testing.assert_equal(output_data.round(4), output_data_restored.round(4))
