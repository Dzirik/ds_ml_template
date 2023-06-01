"""
Tester
"""
from datetime import datetime
from typing import List

import pytest
from pandas import DataFrame

from src.constants.global_constants import FP, P
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.group_df_blocks_by_time_transformer import GroupDFBlocksByTimeTransformer
from src.utils.envs import Envs

ATTR_TIME = "DATETIME"
ATTRIBUTES = [ATTR_TIME, "C1", "C2", "C3"]
GROUP_ATTRIBUTES_1 = [ATTR_TIME, "C1"]
GROUP_ATTRIBUTES_2 = [ATTR_TIME, "C1", "C2"]
GROUP_ATTRIBUTES_3 = [ATTR_TIME, "C1", "C2", "C3"]
DATA = [
    [datetime(2021, 10, 1, 12, 23, 44), 1, 2, 3],
    [datetime(2021, 10, 1, 13, 33, 44), 1, 2, 3],
    [datetime(2021, 10, 1, 14, 43, 44), 1, 2, 3],
    [datetime(2021, 10, 2, 15, 53, 44), 10, 20, 30],
    [datetime(2021, 10, 2, 16, 3, 44), 10, 20, 30],
    [datetime(2021, 10, 2, 17, 13, 44), 10, 20, 30],
    [datetime(2021, 10, 3, 18, 23, 44), 1, 10, 100],
    [datetime(2021, 10, 3, 19, 33, 44), 2, 20, 200],
    [datetime(2021, 10, 3, 20, 43, 44), 3, 30, 300]
]
DATA_FIRST_LAST = [
    [datetime(2021, 10, 1, 12, 23, 44), 3],
    [datetime(2021, 10, 1, 13, 33, 44), 1],
    [datetime(2021, 10, 1, 14, 43, 44), 2],
    [datetime(2021, 10, 2, 15, 53, 44), 10],
    [datetime(2021, 10, 2, 16, 3, 44), 20],
    [datetime(2021, 10, 2, 17, 13, 44), 30],
    [datetime(2021, 10, 3, 18, 23, 44), 300],
    [datetime(2021, 10, 3, 19, 33, 44), 200],
    [datetime(2021, 10, 3, 20, 43, 44), 100]
]
DF = DataFrame(DATA, columns=ATTRIBUTES)
DF_1 = DF[GROUP_ATTRIBUTES_1]
DF_2 = DF[GROUP_ATTRIBUTES_2]
DF_3 = DF[GROUP_ATTRIBUTES_3]
DF_4 = DataFrame(DATA_FIRST_LAST, columns=GROUP_ATTRIBUTES_1)
CORRECT_DF_1_MIN = [1, 10, 1]
CORRECT_DF_1_MAX = [1, 10, 3]
CORRECT_DF_1_MEAN = [1, 10, 2]
CORRECT_DF_1_SUM = [3, 30, 6]
CORRECT_DF_2_MIN = [1, 10, 1]
CORRECT_DF_2_MAX = [2, 20, 30]
CORRECT_DF_2_MEAN = [1.5, 15, 11]
CORRECT_DF_2_SUM = [9, 90, 66]
CORRECT_DF_3_MIN = [1, 10, 1]
CORRECT_DF_3_MAX = [3, 30, 300]
CORRECT_DF_3_MEAN = [2, 20, 74]
CORRECT_DF_4_FIRST = [3, 10, 300]
CORRECT_DF_4_LAST = [2, 30, 100]
CORRECT_DF_3_STD = [0.816496580927726, 8.16496580927726, 101.17641358867523]


def _do_transformation(type_of_method: str, df: DataFrame, attr_time: str, window_length: str, grouping_fun: str) \
        -> DataFrame:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param df: DataFrame. Dataframe to be grouped.
    :param attr_time: str. Name of the attribute to be grouped by.
    :param window_length: str. Window length string format from here
           https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.floor.html
    :param grouping_fun: str. Function to be performed.
    """
    transformer = GroupDFBlocksByTimeTransformer()
    if type_of_method == FP:
        df_out = transformer.fit_predict(df, attr_time, window_length, grouping_fun)
    elif type_of_method == P:
        transformer.fit(df, attr_time, window_length, grouping_fun)
        df_out = transformer.predict()
    else:
        raise NoProperOptionInIf(transformer.get_class_info().class_name)
    return df_out


@pytest.mark.parametrize("type_of_method, df, attr_time, window_length, grouping_fun, correct_list",
                         [
                             (FP, DF_1, ATTR_TIME, "1d", "min", CORRECT_DF_1_MIN),
                             (P, DF_1, ATTR_TIME, "1d", "min", CORRECT_DF_1_MIN),
                             (FP, DF_1, ATTR_TIME, "1d", "max", CORRECT_DF_1_MAX),
                             (P, DF_1, ATTR_TIME, "1d", "max", CORRECT_DF_1_MAX),
                             (FP, DF_1, ATTR_TIME, "1d", "mean", CORRECT_DF_1_MEAN),
                             (P, DF_1, ATTR_TIME, "1d", "mean", CORRECT_DF_1_MEAN),
                             (FP, DF_1, ATTR_TIME, "1d", "min", CORRECT_DF_1_MIN),
                             (P, DF_1, ATTR_TIME, "1d", "min", CORRECT_DF_1_MIN),
                             (FP, DF_1, ATTR_TIME, "1d", "sum", CORRECT_DF_1_SUM),
                             (P, DF_1, ATTR_TIME, "1d", "sum", CORRECT_DF_1_SUM),
                             (FP, DF_2, ATTR_TIME, "1d", "max", CORRECT_DF_2_MAX),
                             (P, DF_2, ATTR_TIME, "1d", "max", CORRECT_DF_2_MAX),
                             (FP, DF_2, ATTR_TIME, "1d", "mean", CORRECT_DF_2_MEAN),
                             (P, DF_2, ATTR_TIME, "1d", "mean", CORRECT_DF_2_MEAN),
                             (FP, DF_2, ATTR_TIME, "1d", "sum", CORRECT_DF_2_SUM),
                             (P, DF_2, ATTR_TIME, "1d", "sum", CORRECT_DF_2_SUM),
                             (FP, DF_3, ATTR_TIME, "1d", "min", CORRECT_DF_3_MIN),
                             (P, DF_3, ATTR_TIME, "1d", "min", CORRECT_DF_3_MIN),
                             (FP, DF_3, ATTR_TIME, "1d", "max", CORRECT_DF_3_MAX),
                             (P, DF_3, ATTR_TIME, "1d", "max", CORRECT_DF_3_MAX),
                             (FP, DF_3, ATTR_TIME, "1d", "mean", CORRECT_DF_3_MEAN),
                             (P, DF_3, ATTR_TIME, "1d", "mean", CORRECT_DF_3_MEAN),
                             (FP, DF_3, ATTR_TIME, "1d", "std", CORRECT_DF_3_STD),
                             (P, DF_3, ATTR_TIME, "1d", "std", CORRECT_DF_3_STD),
                             (FP, DF_4, ATTR_TIME, "1d", "first", CORRECT_DF_4_FIRST),
                             (P, DF_4, ATTR_TIME, "1d", "first", CORRECT_DF_4_FIRST),
                             (FP, DF_4, ATTR_TIME, "1d", "last", CORRECT_DF_4_LAST),
                             (P, DF_4, ATTR_TIME, "1d", "last", CORRECT_DF_4_LAST)
                         ])
def test_transformation(type_of_method: str, df: DataFrame, attr_time: str, window_length: str, grouping_fun: str,
                        correct_list: List[float]) -> None:
    """
    Tests the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param df: DataFrame. Dataframe to be grouped.
    :param attr_time: str. Name of the attribute to be grouped by.
    :param window_length: str. Window length string format from here
           https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.floor.html
    :param grouping_fun: str. Function to be performed. Currently available "mean", "max", "min".
    :param correct_list: List[float]. List of correct values.
    """
    out = _do_transformation(type_of_method, df, attr_time, window_length, grouping_fun)
    assert list(out[grouping_fun.upper()]) == correct_list


def test_incorrect_input() -> None:
    """
    Tests exception raise situation when there is no option selected.
    """
    env = Envs()
    env.set_running_unit_tests()
    transformer = GroupDFBlocksByTimeTransformer()
    with pytest.raises(NoProperOptionInIf):
        transformer.fit(DF_1, "DATETIME", "1d", "wrong_function_name")


def test_none_inverse() -> None:
    """
    Test none in inverse
    """
    transformer = GroupDFBlocksByTimeTransformer()
    assert transformer.inverse() is None
