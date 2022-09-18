"""
Tester

Testing and comparison can be found inf notebooks/documentation/time_series_windows_documentation.py
"""
from typing import Any, Union, Tuple

import pytest
from numpy import array, array_equal, ndarray, dtype
from pandas import DataFrame

from src.exceptions.development_exception import NoProperOptionInIf
from src.constants.global_constants import FP, P
from src.transformations.time_series_windows import TimeSeriesWindowsDummy, TimeSeriesWindowsNumpy, \
    TimeSeriesWindowsPandas


def generate_np_array(n: int, p: int = 1) -> ndarray[Any, dtype[Any]]:
    """
    Generates np array.
    :param n: int.
    :param p: int.
    :return: ndarray[Any, dtype[Any]].
    """
    return array([[x / 10 for x in range(0, i * 10 * n, i * 10)] for i in range(1, p + 1)]).transpose()


def generate_df(n: int, p: int = 1) -> DataFrame:
    """
    Generates data frame.
    :param n: int.
    :param p: int.
    """
    return DataFrame(generate_np_array(n, p))


def _transform(type_of_method: str, transformer_class: Any, data: Union[DataFrame, ndarray[Any, dtype[Any]]], \
               input_window_length: int, output_window_len: int, shift: int) \
        -> Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    """
    Transforms the data.
    :param type_of_method: Type of transformation ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param transformer_class: Any. One of three transformers.
    :param data: Union[DataFrame, ndarray[Any, dtype[Any]]].
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int.
    :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]
    """
    transformer = transformer_class()
    if type_of_method == FP:
        X, Y = transformer.fit_predict(data, input_window_length, output_window_len, shift)
    elif type_of_method == P:
        X, Y = transformer.predict(data, input_window_length, output_window_len, shift)
    else:
        raise NoProperOptionInIf(transformer.get_class_info().class_name)
    return X, Y


@pytest.mark.parametrize("type_of_method, n, p, input_window_length, output_window_len, shift",
                         [
                             (FP, 10, 1, 0, 1, 1), (P, 10, 1, 0, 1, 1),
                             (FP, 10, 1, 1, 0, 1), (P, 10, 1, 1, 0, 1),
                             (FP, 10, 1, 1, 1, 1), (P, 10, 1, 1, 1, 1),
                             (FP, 10, 2, 1, 1, 1), (P, 10, 2, 1, 1, 1),
                             (FP, 10, 2, 1, 2, 1), (P, 10, 2, 1, 2, 1),
                             (FP, 10, 2, 2, 1, 1), (P, 10, 2, 2, 1, 1),
                             (FP, 10, 2, 3, 2, 1), (P, 10, 2, 3, 2, 1),
                             (FP, 100, 10, 5, 4, 3), (P, 100, 10, 5, 4, 3)
                         ])
def test_sum_to_one(type_of_method: str, n: int, p: int, input_window_length: int, output_window_len: int,
                    shift: int) -> None:
    """
    Tests sum to 1 around the state.
    :param type_of_method: Type of transformation ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param n: int. Number of observations.
    :param p: int. Number of attributes.
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
    """
    # pylint: disable=invalid-name
    ts_array = generate_np_array(n, p)
    ts_df = generate_df(n, p)
    X_d, Y_d = _transform(
        type_of_method, TimeSeriesWindowsDummy, ts_array, input_window_length, output_window_len, shift
    )
    X_n, Y_n = _transform(
        type_of_method, TimeSeriesWindowsNumpy, ts_array, input_window_length, output_window_len, shift
    )
    X_p, Y_p = _transform(
        type_of_method, TimeSeriesWindowsPandas, ts_df, input_window_length, output_window_len, shift
    )
    assert array_equal(X_d, X_n) and array_equal(Y_d, Y_n) and \
           array_equal(X_d, X_p) and array_equal(Y_d, Y_p)
    # pylint: enable=invalid-name
