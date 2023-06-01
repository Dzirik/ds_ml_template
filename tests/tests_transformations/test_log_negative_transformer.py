"""
Tester
"""

from typing import Tuple, Any

import pytest
from numpy import array, ndarray, dtype, double, testing

from src.constants.global_constants import FP, P
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.log_negative_transformer import LogNegativeTransformer

DATA: ndarray[Any, dtype[double]] = array([[-2 ** (10 - i) + 1, -2 ** i + 1] for i in list(range(11))] +
                                          [[2 ** i - 1, 2 ** (10 - i) - 1] for i in list(range(11))])
OUTPUT_CORRECT: ndarray[Any, dtype[double]] = array([[float(-(10 - i)), float(-i)] for i in list(range(11))] +
                                                    [[float(i), float(10 - i)] for i in list(range(11))])


def _do_transformation(type_of_method: str, data: ndarray[Any, dtype[double]], base: float) -> \
        Tuple[ndarray[Any, dtype[double]], LogNegativeTransformer]:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    :param base: float. Base of the logarithm.
    :return: Tuple[ndarray[Any, dtype[double]], LogTransformer].
    """
    transformer = LogNegativeTransformer()
    if type_of_method == FP:
        transformed_data = transformer.fit_predict(data, base)
    elif type_of_method == P:
        transformer.fit(data, base)
        transformed_data = transformer.predict(data)
    else:
        raise NoProperOptionInIf

    return transformed_data, transformer


@pytest.mark.parametrize("type_of_method, data, base, correct_output",
                         [
                             (FP, DATA, 2., OUTPUT_CORRECT),
                             (P, DATA, 2., OUTPUT_CORRECT),
                         ])
def test_calculation(type_of_method: str, data: ndarray[Any, dtype[double]], base: float, \
                     correct_output: ndarray[Any, dtype[double]]) -> None:
    """
    Tests that the logarithms are being correctly calculated.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    :param base: float. Base of the logarithm.
    :param correct_output: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    """
    data_out, _ = _do_transformation(type_of_method, data, base)
    print(data_out)
    print(OUTPUT_CORRECT)
    assert (data_out - correct_output).sum().sum() == 0
    # testing.assert_equal(data_out.round(4), correct_output.round(4))


@pytest.mark.parametrize("type_of_method, data, base",
                         [
                             (FP, DATA, 2.),
                             (P, DATA, 2.)
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


@pytest.mark.parametrize("type_of_method, data_input, base",
                         [
                             (FP, DATA, 2),
                             (P, DATA, 2)
                         ])
def test_restoration(type_of_method: str, data_input: ndarray[Any, dtype[double]], base: float) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data_input: ndarray[Any, dtype[double]].
    :param base: float. Base of the logarithm.
    """
    output_data, transformer = _do_transformation(type_of_method, data_input, base)

    transformer_restored = LogNegativeTransformer()
    transformer_restored.restore_from_params(transformer.get_params())
    output_data_restored = transformer_restored.predict(data_input)

    testing.assert_equal(output_data.round(4), output_data_restored.round(4))
