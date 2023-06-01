"""
Rules for anomaly detection rules model.

Documented in the notebook.
"""

from abc import abstractmethod
from typing import Any, NamedTuple, Dict

from numpy import ndarray, dtype, double, sign

from src.exceptions.data_exception import MismatchedDimension
from src.exceptions.development_exception import NoProperOptionInIf
from src.utils.meta_class import MetaClass, RULE_TYPE_NAME


class DetectionRuleParams(NamedTuple):
    """
    Named tuple for storing data for one rule.
    - name: str. Name of the rule.
    - window_len: int. Exact number of residuals needed for the rule.
    - params: Dict[str, Any]. Dictionary of parameters.
    """
    name: str
    window_len: int
    params: Dict[str, Any]


class BaseRule(MetaClass):  # type:ignore
    """
    Base class for rules.
    """

    def __init__(self, class_name: str, window_len: int) -> None:
        MetaClass.__init__(self, class_type=RULE_TYPE_NAME, class_name=class_name)
        self._window_len = window_len

    @abstractmethod
    def apply(self, residuals: ndarray[Any, dtype[double]]) -> int:
        """
        Applies the rule to the residuals.
        :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
        :return: int. No anomaly = 0, is anomaly = 1.
        """

    @staticmethod
    def is_one_sign(residuals: ndarray[Any, dtype[double]]) -> bool:
        """
        Tests if all residuals are of the same sign. If any zero, then false.
        :param residuals: ndarray[Any, dtype[double]]. Array to be tested.
        :return: bool. True if on one side.
        """
        return bool(abs(sum(sign(residuals))) == len(residuals))

    @staticmethod
    def is_positive(residuals: ndarray[Any, dtype[double]]) -> bool:
        """
        Tests if all residuals are positive. If any zero, then false.
        :param residuals: ndarray[Any, dtype[double]]. Array to be tested.
        :return: bool. True if all above.
        """
        return sum(sign(residuals)) == len(residuals)

    @staticmethod
    def is_negative(residuals: ndarray[Any, dtype[double]]) -> bool:
        """
        Tests if all residuals are negative. If any zero, then false.
        :param residuals: ndarray[Any, dtype[double]]. Array to be tested.
        :return: bool. True if all below.
        """
        return sum(sign(residuals)) == - len(residuals)

    def get_window_len(self) -> int:
        """
        Gets the window len.
        :return: int.
        """
        return self._window_len

    def _test_dimension(self, residuals: ndarray[Any, dtype[double]]) -> None:
        """
        Tests the dimension of residuals.
        :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
        """
        if residuals.shape != (self._window_len,):
            raise MismatchedDimension


class NoAnomalyRule(BaseRule):
    """
    Class for testing base class methods separately.
    """

    def __init__(self) -> None:
        BaseRule.__init__(self, class_name="no anomaly", window_len=4)

    def apply(self, residuals: ndarray[Any, dtype[double]]) -> int:
        """
        Applies the rule to the residuals.
        :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
        :return: int. No anomaly = 0, is anomaly = 1.
        """
        return 0


# pylint: disable=arguments-differ
class BaseWecoRule(BaseRule):
    """
    Base class for weco rules.
    """

    def __init__(self, class_name: str, window_len: int) -> None:
        BaseRule.__init__(self, class_name=class_name, window_len=window_len)

    def _evaluate_rule(self, absolute_distance_status: bool, residuals: ndarray[Any, dtype[double]], side: str) \
            -> int:
        """
        Evaluates the rule
        :param absolute_distance_status: bool. If enough observations in the same distance range.
        :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
        :param side: str. "both", "positive", "negative".
        :return: int. No anomaly = 0, is anomaly = 1.
        """
        if side == "positive":
            return int(self.is_positive(residuals) and absolute_distance_status)
        if side == "negative":
            return int(self.is_negative(residuals) and absolute_distance_status)
        if side == "both":
            return int(self.is_one_sign(residuals) and absolute_distance_status)
        raise NoProperOptionInIf


class FirstWecoRule(BaseWecoRule):
    """
    First WECO rule.
    """

    def __init__(self) -> None:
        BaseWecoRule.__init__(self, class_name="FirstWecoRule", window_len=1)

    def apply(self, residuals: ndarray[Any, dtype[double]], std_est: float, side: str = "both",  # type:ignore
              alpha: float = 1.) -> int:
        """
        Applies the rule to the residuals.
        :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
        :param std_est: float. Std of all residuals.
        :param side: str. "both", "positive", "negative".
        :param alpha: float. Multiplier for std, positive number. Takes absolute value.
        :return: int. No anomaly = 0, is anomaly = 1.
        """
        self._test_dimension(residuals)

        alpha = abs(alpha)
        absolute_distance_status = bool(abs(residuals) > 3 * alpha * std_est)

        return self._evaluate_rule(absolute_distance_status, residuals, side)


class SecondWecoRule(BaseWecoRule):
    """
    Second WECO rule.
    """

    def __init__(self) -> None:
        BaseWecoRule.__init__(self, class_name="SecondWecoRule", window_len=3)

    def apply(self, residuals: ndarray[Any, dtype[double]], std_est: float, side: str = "both",  # type:ignore
              alpha: float = 1.) -> int:
        """
        Applies the rule to the residuals.
        :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
        :param std_est: float. Std of all residuals.
        :param side: str. "both", "positive", "negative".
        :param alpha: float. Multiplier for std, positive number. Takes absolute value.
        :return: int. No anomaly = 0, is anomaly = 1.
        """
        self._test_dimension(residuals)

        alpha = abs(alpha)
        absolute_distance_status = sum(abs(residuals) > 2 * alpha * std_est) >= 2

        return self._evaluate_rule(absolute_distance_status, residuals, side)


class ThirdWecoRule(BaseWecoRule):
    """
    Third WECO rule.
    """

    def __init__(self) -> None:
        BaseWecoRule.__init__(self, class_name="ThirdWecoRule", window_len=5)

    def apply(self, residuals: ndarray[Any, dtype[double]], std_est: float, side: str = "both",  # type:ignore
              alpha: float = 1.) -> int:
        """
        Applies the rule to the residuals.
        :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
        :param std_est: float. Std of all residuals.
        :param side: str. "both", "positive", "negative".
        :param alpha: float. Multiplier for std, positive number. Takes absolute value.
        :return: int. No anomaly = 0, is anomaly = 1.
        """
        self._test_dimension(residuals)

        alpha = abs(alpha)
        absolute_distance_status = sum(abs(residuals) > alpha * std_est) >= 4

        return self._evaluate_rule(absolute_distance_status, residuals, side)


class FourthWecoRule(BaseWecoRule):
    """
    Fourth WECO rule.
    """

    def __init__(self) -> None:
        BaseWecoRule.__init__(self, class_name="FourthWecoRule", window_len=8)

    # pylint: disable=unused-argument
    def apply(self, residuals: ndarray[Any, dtype[double]], std_est: float, side: str = "both",  # type:ignore
              alpha: float = 1.) -> int:
        """
        Applies the rule to the residuals.
        :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
        :param std_est: float. Std of all residuals.
        :param side: str. "both", "positive", "negative".
        :param alpha: float. Multiplier for std, positive number. Takes absolute value.
        :return: int. No anomaly = 0, is anomaly = 1.
        """
        self._test_dimension(residuals)

        absolute_distance_status = True

        return self._evaluate_rule(absolute_distance_status, residuals, side)
    # pylint: enable=unused-argument
# pylint: enable=arguments-differ
