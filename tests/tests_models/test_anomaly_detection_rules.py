"""
Tester

Tests are just overall, rules are properly tested in anomaly_detection_rules.py.
"""

from typing import Any

import pytest
from numpy import ndarray, dtype, double, array

from src.exceptions.data_exception import MismatchedDimension
from src.exceptions.development_exception import NoProperOptionInIf
from src.models.anomaly_detection_rules import FirstWecoRule, SecondWecoRule, ThirdWecoRule, \
    FourthWecoRule, NoAnomalyRule

NO_ANOMALY_RULE = NoAnomalyRule()


@pytest.mark.parametrize("residuals, is_one_side, is_above, is_below",
                         [
                             (array([1., 2., 3.]), True, True, False),
                             (-1. * array([1., 2., 3.]), True, False, True),
                             (array([1., -2., 3]), False, False, False),
                             (array([0., -2., 3]), False, False, False),
                             (array([0., 0., 0.]), False, False, False)
                         ])
def test_one_side_above_below(residuals: ndarray[Any, dtype[double]], is_one_side: bool, is_above: bool, \
                              is_below: bool) -> None:
    """
    Tests one side, above, bellow in base rule comparing zero level as default.
    :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
    :param is_one_side: bool.
    :param is_above: bool.
    :param is_below: bool.
    """
    assert NO_ANOMALY_RULE.is_one_sign(residuals) == is_one_side and \
           NO_ANOMALY_RULE.is_positive(residuals) == is_above and \
           NO_ANOMALY_RULE.is_negative(residuals) == is_below


def test_return_of_window_len() -> None:
    """
    Tests correct return of window length.b
    """
    assert NO_ANOMALY_RULE.get_window_len() == 4


@pytest.mark.parametrize("weco_class, residuals, std_est, side, alpha, correct_output",
                         [
                             (FirstWecoRule, array([10.]), 1., "both", 1., 1),
                             (FirstWecoRule, array([10.]), 1., "positive", 1., 1),
                             (FirstWecoRule, array([10.]), 1., "negative", 1., 0),
                             (FirstWecoRule, array([-10.]), 1., "both", 2., 1),
                             (FirstWecoRule, array([-10.]), 1., "positive", 2., 0),
                             (FirstWecoRule, array([-10.]), 1., "negative", 2., 1),
                             (FirstWecoRule, array([2.]), 1., "both", 1., 0),
                             (FirstWecoRule, array([2.]), 1., "positive", 1., 0),
                             (FirstWecoRule, array([2.]), 1., "negative", 1., 0),
                             (FirstWecoRule, array([-2.]), 1., "both", 2., 0),
                             (FirstWecoRule, array([-2.]), 1., "positive", 2., 0),
                             (FirstWecoRule, array([-2.]), 1., "negative", 2., 0),
                             (SecondWecoRule, array([5., 1., 6.]), 1., "both", 1., 1),
                             (SecondWecoRule, array([5., 1., 6.]), 1., "positive", 1., 1),
                             (SecondWecoRule, array([5., 1., 6.]), 1., "negative", 1., 0),
                             (SecondWecoRule, -1 * array([5., 1., 6.]), 1., "both", 1.5, 1),
                             (SecondWecoRule, -1 * array([5., 1., 6.]), 1., "positive", 1.5, 0),
                             (SecondWecoRule, -1 * array([5., 1., 6.]), 1., "negative", 1.5, 1),
                             (SecondWecoRule, array([2., 2., 1.]), 1., "both", 1., 0),
                             (SecondWecoRule, array([3., 3., 1.]), 1., "both", 1., 1),
                             (SecondWecoRule, array([3., 3., -1.]), 1., "both", 1., 0),
                             (SecondWecoRule, array([2., 2., 1.]), 1., "positive", 1., 0),
                             (SecondWecoRule, array([3., 3., 1.]), 1., "positive", 1., 1),
                             (SecondWecoRule, array([3., 3., -1.]), 1., "positive", 1., 0),
                             (SecondWecoRule, -1 * array([2., 2., 1.]), 1., "negative", 1., 0),
                             (SecondWecoRule, -1 * array([3., 3., 1.]), 1., "negative", 1., 1),
                             (SecondWecoRule, -1 * array([3., 3., -1.]), 1., "negative", 1., 0),
                             (SecondWecoRule, array([10., -10., 10.]), 1., "both", 1., 0),
                             (SecondWecoRule, array([10., -10., 10.]), 1., "positive", 1., 0),
                             (SecondWecoRule, -1 * array([10., -10., 10.]), 1., "negative", 1., 0),
                             (ThirdWecoRule, array([4., 3., 1., 5., 10.]), 1., "both", 1., 1),
                             (ThirdWecoRule, array([4., 3., 1., 5., 10.]), 1., "positive", 1., 1),
                             (ThirdWecoRule, array([4., 3., 1., 5., 10.]), 1., "negative", 1., 0),
                             (ThirdWecoRule, -1 * array([4., 3., 1., 5., 10.]), 1., "both", 1., 1),
                             (ThirdWecoRule, -1 * array([4., 3., 1., 5., 10.]), 1., "positive", 1., 0),
                             (ThirdWecoRule, -1 * array([4., 3., 1., 5., 10.]), 1., "negative", 1., 1),
                             (ThirdWecoRule, array(5 * [1.]), 1., "both", 1., 0),
                             (ThirdWecoRule, array(5 * [1.]), 1., "positive", 1., 0),
                             (ThirdWecoRule, array(5 * [1.]), 1., "positive", 1., 0),
                             (ThirdWecoRule, array([10., 10., 0., 10., 10.]), 2., "both", 1., 0),
                             (ThirdWecoRule, array([10., 10., -1., 10., 10.]), 2., "both", 1., 0),
                             (ThirdWecoRule, array([10., 10., 0., 10., 10.]), 2., "positive", 1., 0),
                             (ThirdWecoRule, array([10., 10., -1., 10., 10.]), 2., "positive", 1., 0),
                             (ThirdWecoRule, -1 * array([10., 10., 0., 10., 10.]), 2., "negative", 1., 0),
                             (ThirdWecoRule, -1 * array([10., 10., -1., 10., 10.]), 2., "negative", 1., 0),
                             (ThirdWecoRule, array(2 * [-10.] + 3 * [10.]), 1., "both", 1., 0),
                             (ThirdWecoRule, array(2 * [-10.] + 3 * [10.]), 1., "positive", 1., 0),
                             (ThirdWecoRule, -1 * array(2 * [-10.] + 3 * [10.]), 1., "negative", 1., 0),
                             (FourthWecoRule, array(8 * [1.]), 2., "both", 1., 1),
                             (FourthWecoRule, array(8 * [1.]), 2., "positive", 1., 1),
                             (FourthWecoRule, array(8 * [1.]), 2., "negative", 1., 0),
                             (FourthWecoRule, -1 * array(8 * [1.]), 2., "both", 1., 1),
                             (FourthWecoRule, -1 * array(8 * [1.]), 2., "positive", 1., 0),
                             (FourthWecoRule, -1 * array(8 * [1.]), 2., "negative", 1., 1),
                             (FourthWecoRule, array([1., 1., 1., 1., 0., 1., 1., 1.]), 10., "both", 10., 0),
                             (FourthWecoRule, array([1., 1., 1., 1., -1., 1., 1., 1.]), 10., "both", 10., 0),
                             (FourthWecoRule, array([1., 1., 1., 1., 0., 1., 1., 1.]), 10., "positive", 10., 0),
                             (FourthWecoRule, array([1., 1., 1., 1., -1., 1., 1., 1.]), 10., "positive", 10., 0),
                             (FourthWecoRule, -1 * array([1., 1., 1., 1., 0., 1., 1., 1.]), 10., "negative", 10., 0),
                             (FourthWecoRule, -1. * array([1., 1., 1., 1., -1., 1., 1., 1.]), 10., "negative", 10., 0),
                         ])
def test_weco_rules(weco_class: Any, residuals: ndarray[Any, dtype[double]], std_est: float, side: str, alpha: float, \
                    correct_output: int) -> None:
    """
    Tests the rule.
    :param weco_class. Any. Class of weco rule.
    :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
    :param std_est: float. Std of all residuals.
    :param side: str. "both", "positive", "negative".
    :param alpha: float. Multiplier for std.
    :param correct_output: int. 0 - no anomaly, 1 - is anomaly.
    :return: int. No anomaly = 0, is anomaly = 1.
    """
    assert weco_class().apply(residuals, std_est, side, alpha) == correct_output


@pytest.mark.parametrize("weco_class, residuals",
                         [
                             (FirstWecoRule, array([1.])),
                             (SecondWecoRule, array([1., 1., 1.])),
                             (ThirdWecoRule, array(5 * [1.])),
                             (FourthWecoRule, array(8 * [1.]))
                         ])
def test_no_proper_side_in_weco_rules(weco_class: Any, residuals: ndarray[Any, dtype[double]]) -> None:
    """
    Tests exception.
    :param weco_class. Any. Class of weco rule.
    :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
    """
    with pytest.raises(NoProperOptionInIf):
        weco_class().apply(residuals, 1., "wrong_side", 1.)


@pytest.mark.parametrize("weco_class, residuals",
                         [
                             (FirstWecoRule, array([1., 1.])),
                             (FirstWecoRule, array([[1.], [1.]])),
                             (SecondWecoRule, array([1., 1.])),
                             (SecondWecoRule, array([[1.], [1.]])),
                             (ThirdWecoRule, array([1., 1.])),
                             (ThirdWecoRule, array([[1.], [1.]])),
                             (FourthWecoRule, array([1., 1.])),
                             (FourthWecoRule, array([[1.], [1.]])),
                         ])
def test_raise_no_proper_dimension_in_weco_rules(weco_class: Any, residuals: ndarray[Any, dtype[double]]) -> None:
    """
    Tests exception.
    :param weco_class. Any. Class of weco rule.
    :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
    """
    with pytest.raises(MismatchedDimension):
        weco_class().apply(residuals, 1., "wrong_side", 1.)


def test_no_anomaly_rule() -> None:
    """
    Tests test no anomaly rule.
    """
    rule = NoAnomalyRule()
    assert rule.apply(array([1.])) == 0
