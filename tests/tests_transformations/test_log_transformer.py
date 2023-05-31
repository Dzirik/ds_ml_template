"""
Tester
"""

from typing import List, Tuple, Any

import pytest
from numpy import array, ndarray, dtype, double, e, testing

from src.constants.global_constants import FP, P
from src.exceptions.data_exception import IncorrectDataStructure
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.log_transformer import LogTransformer


def create_exp_array(bases: List[float]) -> ndarray[Any, dtype[double]]:
    """
    Creates a testing array.
    :param bases: List[float]. List of bases - attributes.
    :return:
    """
    n = 6
    data = []
    for i in range(n):
        data.append([b ** i - 1 for b in bases])
    return array(data)


ARRAY_BASES: ndarray[Any, dtype[double]] = create_exp_array([2., e, 6., 10.])
BASE_CORRECT: ndarray[Any, dtype[double]] = array([[0.], [1.], [2.], [3.], [4.], [5.]])


def _do_transformation(type_of_method: str, data: ndarray[Any, dtype[double]], base: float) -> \
        Tuple[ndarray[Any, dtype[double]], LogTransformer]:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    :param base: float. Base of the logarithm.
    :return: Tuple[ndarray[Any, dtype[double]], LogTransformer].
    """
    transformer = LogTransformer()
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
                             (FP, ARRAY_BASES[:, 0].reshape((ARRAY_BASES.shape[0], 1)), 2., BASE_CORRECT),
                             (P, ARRAY_BASES[:, 0].reshape((ARRAY_BASES.shape[0], 1)), 2., BASE_CORRECT),
                             (FP, ARRAY_BASES[:, 1].reshape((ARRAY_BASES.shape[0], 1)), e, BASE_CORRECT),
                             (P, ARRAY_BASES[:, 1].reshape((ARRAY_BASES.shape[0], 1)), e, BASE_CORRECT),
                             (FP, ARRAY_BASES[:, 2].reshape((ARRAY_BASES.shape[0], 1)), 6., BASE_CORRECT),
                             (P, ARRAY_BASES[:, 2].reshape((ARRAY_BASES.shape[0], 1)), 6., BASE_CORRECT),
                             (FP, ARRAY_BASES[:, 3].reshape((ARRAY_BASES.shape[0], 1)), 10., BASE_CORRECT),
                             (P, ARRAY_BASES[:, 3].reshape((ARRAY_BASES.shape[0], 1)), 10., BASE_CORRECT)
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
    testing.assert_equal(data_out.round(4), correct_output.round(4))


@pytest.mark.parametrize("type_of_method, data, base",
                         [
                             (FP, ARRAY_BASES, 2.),
                             (P, ARRAY_BASES, 2.),
                             (FP, ARRAY_BASES, e),
                             (P, ARRAY_BASES, e),
                             (FP, ARRAY_BASES, 6.),
                             (P, ARRAY_BASES, 6.),
                             (FP, ARRAY_BASES, 10.),
                             (P, ARRAY_BASES, 10.)
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


@pytest.mark.parametrize("type_of_method, data",
                         [
                             (FP, array([[-1.]])),
                             (P, array([[-1]])),
                             (FP, array([[1., 3.], [3., -4.]])),
                             (P, array([[1., 3.], [3., -4.]])),
                             (FP, array([[1., -1.]])),
                             (P, array([[1., -1.]])),
                             (FP, array([[1., -3.]])),
                             (P, array([[1., -3.]]))
                         ])
def test_negative_value_error(type_of_method: str, data: ndarray[Any, dtype[double]]) -> None:
    """
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    """
    with pytest.raises(IncorrectDataStructure):
        _, _ = _do_transformation(type_of_method, data, 10.)


@pytest.mark.parametrize("type_of_method, data_input, base",
                         [
                             (FP, ARRAY_BASES, e),
                             (P, ARRAY_BASES, e),
                             (FP, ARRAY_BASES, 10.),
                             (P, ARRAY_BASES, 10.)
                         ])
def test_restoration(type_of_method: str, data_input: ndarray[Any, dtype[double]], base: float) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data_input: ndarray[Any, dtype[double]].
    :param base: float. Base of the logarithm.
    """
    output_data, transformer = _do_transformation(type_of_method, data_input, base)

    transformer_restored = LogTransformer()
    transformer_restored.restore_from_params(transformer.get_params())
    output_data_restored = transformer_restored.predict(data_input)

    testing.assert_equal(output_data.round(4), output_data_restored.round(4))
