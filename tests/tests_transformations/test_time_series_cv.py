"""
Tester

You can see more information in time_series_cv_documentation notebook.

Note: The tests are partially overlaping with test for time_series_windows. But I kept them here as well.
"""
# TODO: Maybe add some tests for bigger variability?

import pytest
from numpy import array

from src.exceptions.data_exception import WrongSorting, MismatchedDimension
from src.transformations.time_series_cv import TimeSeriesCV, create_weather_data, create_sample_data
from src.utils.envs import Envs

# pylint: disable=no-name-in-module


# pylint: enable=no-name-in-module

W_DATE_TIME, W_SPLIT_DATES, W_X_DATA, W_Y_DATA = create_weather_data()
S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, S_Y_DATA, S_DATE_TIME_NOT_SORTED = create_sample_data(31)


@pytest.mark.parametrize("input_window_len, output_window_len, shift, shuffle_data",
                         [
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (2, 2, 1, False),
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, True),
                             (2, 1, 1, True)
                         ])
def test_dimensions_same_input(input_window_len: int, output_window_len: int, shift: int, shuffle_data: bool) -> None:
    """
    Tests if output first dimension of X an Y data is the same.
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
    :param shuffle_data: bool. If to shuffle the data.
    """
    transformer = TimeSeriesCV()
    cross_validation_data = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, S_X_DATA, True,
                                                    input_window_len,
                                                    output_window_len, shift, shuffle_data)
    dim_diffs = 0
    for (X_train, Y_train), (X_test, Y_test) in cross_validation_data:
        dim_diffs += abs(X_train.shape[0] - Y_train.shape[0]) + abs(X_test.shape[0] - Y_test.shape[0])
    assert dim_diffs == 0


@pytest.mark.parametrize("input_window_len, output_window_len, shift, shuffle_data",
                         [
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, True),
                             (2, 1, 1, True)
                         ])
def test_first_y_following_same_input(input_window_len: int, output_window_len: int, shift: int, \
                                      shuffle_data: bool) -> None:
    """
    Tests if the first in Y is the following to X last.
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
    :param shuffle_data: bool. If to shuffle the data.
    """
    transformer = TimeSeriesCV()
    cross_validation_data = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, S_X_DATA, False,
                                                    input_window_len, output_window_len, shift, shuffle_data)
    diffs = 0
    for (X_train, Y_train), (X_test, Y_test) in cross_validation_data:
        for X, Y in zip(X_train, Y_train):
            diffs += abs(Y[0, 0] - (X[-1, 0] + 1)) + abs(Y[0, 0] - (X[-1, 0] + 1))
        for X, Y in zip(X_test, Y_test):
            diffs += abs(Y[0, 0] - (X[-1, 0] + 1)) + abs(Y[0, 0] - (X[-1, 0] + 1))
    assert diffs == 0


@pytest.mark.parametrize("input_window_len, output_window_len, shift, shuffle_data",
                         [
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, True),
                             (2, 1, 1, True)
                         ])
def test_first_y_equal_same_input(input_window_len: int, output_window_len: int, shift: int, \
                                  shuffle_data: bool) -> None:
    """
    Tests if the first in Y is the same
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
    :param shuffle_data: bool. If to shuffle the data.
    """
    transformer = TimeSeriesCV()
    cross_validation_data = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, S_X_DATA, True,
                                                    input_window_len, output_window_len, shift, shuffle_data)
    diffs = 0
    for (X_train, Y_train), (X_test, Y_test) in cross_validation_data:
        for X, Y in zip(X_train, Y_train):
            diffs += abs(Y[0, 0] - X[-1, 0]) + abs(Y[0, 0] - X[-1, 0])
        for X, Y in zip(X_test, Y_test):
            diffs += abs(Y[0, 0] - X[-1, 0]) + abs(Y[0, 0] - X[-1, 0])
    assert diffs == 0


@pytest.mark.parametrize("input_window_len, output_window_len, shift, shuffle_data",
                         [
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (2, 2, 1, False),
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, True),
                             (2, 1, 1, True)
                         ])
def test_dimensions_different_input(input_window_len: int, output_window_len: int, shift: int, shuffle_data: bool) \
        -> None:
    """
    Tests if output first dimension of X an Y data is the same.
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
    :param shuffle_data: bool. If to shuffle the data.
    """
    transformer = TimeSeriesCV()
    cross_validation_data = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, S_Y_DATA, True,
                                                    input_window_len,
                                                    output_window_len, shift, shuffle_data)
    dim_diffs = 0
    for (X_train, Y_train), (X_test, Y_test) in cross_validation_data:
        dim_diffs += abs(X_train.shape[0] - Y_train.shape[0]) + abs(X_test.shape[0] - Y_test.shape[0])
    assert dim_diffs == 0


@pytest.mark.parametrize("input_window_len, output_window_len, shift, shuffle_data",
                         [
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, True),
                             (2, 1, 1, True)
                         ])
def test_first_y_following_different_input(input_window_len: int, output_window_len: int, shift: int, \
                                           shuffle_data: bool) -> None:
    """
    Tests if the first in Y is the following to X last.
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
    :param shuffle_data: bool. If to shuffle the data.
    """
    transformer = TimeSeriesCV()
    cross_validation_data = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, S_Y_DATA, False,
                                                    input_window_len, output_window_len, shift, shuffle_data)
    diffs = 0
    for (X_train, Y_train), (X_test, Y_test) in cross_validation_data:
        for X, Y in zip(X_train, Y_train):
            diffs += abs(Y[0, 0] - 100 * (X[-1, 0] + 1)) + abs(Y[0, 0] - 100 * (X[-1, 0] + 1))
        for X, Y in zip(X_test, Y_test):
            diffs += abs(Y[0, 0] - 100 * (X[-1, 0] + 1)) + abs(Y[0, 0] - 100 * (X[-1, 0] + 1))
    assert diffs == 0


@pytest.mark.parametrize("input_window_len, output_window_len, shift, shuffle_data",
                         [
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, True),
                             (2, 1, 1, True)
                         ])
def test_first_y_equal_different_input(input_window_len: int, output_window_len: int, shift: int, \
                                       shuffle_data: bool) -> None:
    """
    Tests if the first in Y is the same
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
    :param shuffle_data: bool. If to shuffle the data.
    """
    transformer = TimeSeriesCV()
    cross_validation_data = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, S_Y_DATA, True,
                                                    input_window_len, output_window_len, shift, shuffle_data)
    diffs = 0
    for (X_train, Y_train), (X_test, Y_test) in cross_validation_data:
        for X, Y in zip(X_train, Y_train):
            diffs += abs(Y[0, 0] - 100 * X[-1, 0]) + abs(Y[0, 0] - 100 * X[-1, 0])
        for X, Y in zip(X_test, Y_test):
            diffs += abs(Y[0, 0] - 100 * X[-1, 0]) + abs(Y[0, 0] - 100 * X[-1, 0])
    assert diffs == 0


@pytest.mark.parametrize("input_window_len, output_window_len, shift, shuffle_data",
                         [
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, False),
                             (2, 1, 1, False),
                             (1, 1, 1, True),
                             (2, 1, 1, True)
                         ])
def test_connection_y_only(input_window_len: int, output_window_len: int, shift: int, shuffle_data: bool) -> None:
    """
    Tests if the first in Y is the same as X last for taking the Y as well.
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
    :param shuffle_data: bool. If to shuffle the data.
    """
    transformer = TimeSeriesCV()
    cross_validation_data = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, S_X_DATA, True,
                                                    input_window_len, output_window_len, shift, shuffle_data)
    diffs = 0
    for (X_train, Y_train), (X_test, Y_test) in cross_validation_data:
        for X, Y in zip(X_train, Y_train):
            diffs += abs(Y[0, 0] - X[-1, 0]) + abs(Y[0, 0] - X[-1, 0])
        for X, Y in zip(X_test, Y_test):
            diffs += abs(Y[0, 0] - X[-1, 0]) + abs(Y[0, 0] - X[-1, 0])
    assert diffs == 0


def test_not_sorted() -> None:
    """
    Tests exception if date_time not sorted.
    """
    env = Envs()
    env.set_running_unit_tests()
    transformer = TimeSeriesCV()
    with pytest.raises(WrongSorting):
        _ = transformer.fit_predict(S_DATE_TIME_NOT_SORTED, S_SPLIT_DATES, S_X_DATA, S_X_DATA, True, 1, 1, 1, False)


def test_wrong_date_time_data_dim() -> None:
    """
    Tests exception if date_time and X_data do not have the same first dimension.
    """
    env = Envs()
    env.set_running_unit_tests()
    transformer = TimeSeriesCV()
    with pytest.raises(MismatchedDimension):
        print(len(S_DATE_TIME))
        _ = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, array([[1, 2]]), array([[1, 2]]), True, 1, 1, 1, False)


def test_wrong_data_dims() -> None:
    """
    Tests exception if X_data and Y_data do not have the same first dimension.
    """
    env = Envs()
    env.set_running_unit_tests()
    transformer = TimeSeriesCV()
    with pytest.raises(MismatchedDimension):
        _ = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, array([[1, 1]]), True, 1, 1, 1, False)


def test_wrong_x_dim() -> None:
    """
    Tests exception if X_data are not at least two dimensional.
    """
    env = Envs()
    env.set_running_unit_tests()
    transformer = TimeSeriesCV()
    with pytest.raises(MismatchedDimension):
        _ = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, array([1.] * len(S_DATE_TIME)),
                                    S_X_DATA, True, 1, 1, 1, False)


def test_wrong_y_dim() -> None:
    """
    Tests exception if Y_data are not at least two dimensional.
    """
    env = Envs()
    env.set_running_unit_tests()
    transformer = TimeSeriesCV()
    with pytest.raises(MismatchedDimension):
        _ = transformer.fit_predict(S_DATE_TIME, S_SPLIT_DATES, S_X_DATA, array([1.] * len(S_DATE_TIME)), True, 1, 1, 1,
                                    False)


def test_none_inverse() -> None:
    """
    Test none in inverse
    """
    transformer = TimeSeriesCV()
    assert transformer.inverse() is None
