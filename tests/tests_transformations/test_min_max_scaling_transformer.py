"""
Tester
"""
from typing import Any, Tuple

import pytest
from numpy import ndarray, dtype, double, array, testing
from numpy.random import random, seed

from src.constants.global_constants import FP, P
from src.data.anomaly_detection_sample_data import AnomalyDetectionSampleData, ATTRS
from src.exceptions.data_exception import IncorrectDataStructure
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.min_max_scaling_transformer import MinMaxScalingTransformer

INPUT_1: ndarray[Any, dtype[double]] = array([[34500], [23780], [5460], [10570]])
OUTPUT_1: ndarray[Any, dtype[double]] = array([[1.], [0.630854], [0.], [0.175964]])

INPUT_RANGE: Tuple[float, float] = (1., 2.)

OUTPUT_1_RANGE: ndarray[Any, dtype[double]] = array([[2.], [1.6309], [1.], [1.176]])

INPUT_FIT_NULL: ndarray[Any, dtype[double]] = array([[10], [10], [10]])
INPUT_FIT_MATRIX_NULL: ndarray[Any, dtype[double]] = array([[10, 21], [10, 43], [10, 1]])

sample_data = AnomalyDetectionSampleData(abs_values=False)
SAMPLE_DATA_1 = sample_data.generate(50, 123)[ATTRS].to_numpy()
SAMPLE_DATA_2 = sample_data.generate(25, 321)[ATTRS].to_numpy()


def _do_transformation(type_of_method: str, data: ndarray[Any, dtype[double]], single_columns: bool, \
                       feature_range: Tuple[float, float] = (0., 1.)) -> \
        Tuple[ndarray[Any, dtype[double]], MinMaxScalingTransformer]:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    :param single_columns: bool. If to do the transformation separately for single columns or not.
    :param feature_range: Tuple[float, float]. Range for output.
    :return: Tuple[ndarray[Any, dtype[double]], MinMaxScalingTransformer].
    """
    transformer = MinMaxScalingTransformer(feature_range)
    if type_of_method == FP:
        data_output = transformer.fit_predict(data, single_columns)
    elif type_of_method == P:
        transformer.fit(data, single_columns)
        data_output = transformer.predict(data)
    else:
        raise NoProperOptionInIf(transformer.get_class_info().class_name)
    return data_output, transformer


@pytest.mark.parametrize("type_of_method, data_input, single_columns, data_output",
                         [
                             (FP, INPUT_1, True, OUTPUT_1),
                             (P, INPUT_1, True, OUTPUT_1)
                         ])
def test_transformation(type_of_method: str, data_input: ndarray[Any, dtype[double]], single_columns: bool, \
                        data_output: ndarray[Any, dtype[double]]) -> None:
    """
    Tests the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data_input: ndarray[Any, dtype[double]]. 2d array with observations in rows, attributes in columns.
    :param single_columns: bool. If to do the transformation separately for single columns or not.
    :param data_output: ndarray[Any, dtype[double]]. Transformed data.
    """
    output, _ = _do_transformation(type_of_method, data_input, single_columns)
    testing.assert_equal(output.round(4), data_output.round(4))


@pytest.mark.parametrize("type_of_method, data_input, single_columns, data_output, feature_range",
                         [
                             (FP, INPUT_1, True, OUTPUT_1_RANGE, INPUT_RANGE),
                             (P, INPUT_1, True, OUTPUT_1_RANGE, INPUT_RANGE)
                         ])
def test_scaling_with_different_range(type_of_method: str, data_input: ndarray[Any, dtype[double]], \
                                      single_columns: bool, \
                                      data_output: ndarray[Any, dtype[double]], \
                                      feature_range: Tuple[int, int]) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data_input: ndarray[Any, dtype[double]].
    :param single_columns: bool. If to do the transformation separately for single columns or not.
    :param data_output: ndarray[Any, dtype[double]].
    :param feature_range: Tuple[int, int].
    """
    output, _ = _do_transformation(type_of_method, data_input, single_columns, feature_range)
    testing.assert_equal(output.round(4), data_output.round(4))


@pytest.mark.parametrize("type_of_method, feature_range, n, p, seed_number",
                         [
                             (FP, (0., 1.), 20, 2, 9803),
                             (P, (0., 1.), 20, 2, 9803),
                             (FP, (1., 2.), 25, 4, 87),
                             (P, (1., 2.), 25, 4, 87),
                             (FP, (0., 1.), 20, 2, 98),
                             (P, (0., 1.), 20, 2, 98),
                             (FP, (1., 2.), 25, 4, 876523),
                             (P, (1., 2.), 25, 4, 876523)
                         ])
def test_transformation_for_multi_columns(type_of_method: str, feature_range: Tuple[int, int], n: int, p: int, \
                                          seed_number: int) -> None:
    """

    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param feature_range: Tuple[int, int].
    :param n: int. Number of rows.
    :param p: int. Number of columns.
    :param seed_number: int. Seed for random generation.
    """
    seed(seed_number)
    data_one_dim = random(n * p).reshape((n * p, 1)) * 1000
    data_multi_dim = data_one_dim.reshape((n, p))

    output_one_dim, _ = _do_transformation(type_of_method, data_one_dim, False, feature_range)
    output_one_dim_reshaped = output_one_dim.reshape((n, p))

    output_multi_dim, _ = _do_transformation(type_of_method, data_multi_dim, False, feature_range)

    testing.assert_equal(output_one_dim_reshaped.round(4), output_multi_dim.round(4))


@pytest.mark.parametrize("type_of_method, data_input, single_columns",
                         [
                             (FP, INPUT_FIT_NULL, True),
                             (P, INPUT_FIT_NULL, True),
                             (FP, INPUT_FIT_MATRIX_NULL, True),
                             (P, INPUT_FIT_MATRIX_NULL, True),
                             (FP, INPUT_FIT_NULL, False),
                             (P, INPUT_FIT_NULL, False),
                             (FP, INPUT_FIT_MATRIX_NULL, False),
                             (P, INPUT_FIT_MATRIX_NULL, False)
                         ])
def test_null_std(type_of_method: str, data_input: ndarray[Any, dtype[double]], single_columns: bool) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data_input: ndarray[Any, dtype[double]].
    :param single_columns: bool. If to do the transformation separately for single columns or not.
    """
    with pytest.raises(IncorrectDataStructure):
        _do_transformation(type_of_method, data_input, single_columns)


@pytest.mark.parametrize("type_of_method, data_input, single_columns",
                         [
                             (FP, SAMPLE_DATA_1, True),
                             (P, SAMPLE_DATA_1, True),
                             (FP, SAMPLE_DATA_2, True),
                             (P, SAMPLE_DATA_2, True),
                             (FP, SAMPLE_DATA_1, False),
                             (P, SAMPLE_DATA_1, False),
                             (FP, SAMPLE_DATA_2, False),
                             (P, SAMPLE_DATA_2, False)
                         ])
def test_inverse(type_of_method: str, data_input: ndarray[Any, dtype[double]], single_columns: bool) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data_input: ndarray[Any, dtype[double]].
    :param single_columns: bool. If to do the transformation separately for single columns or not.
    """
    output_data, transformer = _do_transformation(type_of_method, data_input, single_columns)
    data_input_restored = transformer.inverse(output_data)
    testing.assert_equal(data_input.round(4), data_input_restored.round(4))


@pytest.mark.parametrize("type_of_method, data_input, single_columns",
                         [
                             (FP, SAMPLE_DATA_1, True),
                             (P, SAMPLE_DATA_1, True),
                             (FP, SAMPLE_DATA_2, True),
                             (P, SAMPLE_DATA_2, True),
                             (FP, SAMPLE_DATA_1, False),
                             (P, SAMPLE_DATA_1, False),
                             (FP, SAMPLE_DATA_2, False),
                             (P, SAMPLE_DATA_2, False)
                         ])
def test_restoration(type_of_method: str, data_input: ndarray[Any, dtype[double]], single_columns: bool) -> None:
    """
    Tests that a data set is properly scaled within a range accordingly to a fitted data set.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data_input: ndarray[Any, dtype[double]].
    :param single_columns: bool. If to do the transformation separately for single columns or not.
    """
    output_data, transformer = _do_transformation(type_of_method, data_input, single_columns)

    transformer_restored = MinMaxScalingTransformer()
    transformer_restored.restore_from_params(transformer.get_params())
    output_data_restored = transformer_restored.predict(data_input)

    testing.assert_equal(output_data.round(4), output_data_restored.round(4))
