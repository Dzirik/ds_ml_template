"""
Tester
"""
from typing import List, Tuple

import pytest
from pandas import DataFrame, Series

from src.constants.global_constants import FP, P
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.group_df_columns_transformer import GroupDFColumnsTransformer
from src.utils.envs import Envs

DATA_1 = [
    [1., 2., 3.],
    [10., 20., 30.],
    [1., 1., 1.],
    [10., 10., 10.]
]
ATTRS_1 = ["NUMBER_1", "NUMBER_2", "NUMBER_3"]
ATTRS_2 = ["NUMBER_1", "NUMBER_2"]
DF_1 = DataFrame(DATA_1, columns=ATTRS_1)
CORRECT_DF_1_MEAN = [2., 20., 1., 10.]
CORRECT_DF_1_STD = [1., 10., 0., 0.]
CORRECT_DF_1_SUM = [6., 60., 3., 30.]
CORRECT_DF_2_SUM = [3., 30., 2., 20.]
CORRECT_NAN = ["", "", "", ""]
ATTR_MINUS = ["NUMBER_3"]
CORRECT_MINUS = [-3., -30., -1., -10.]


def generate_list_data_past_and_future(shift: int) -> Tuple[DataFrame, Series, Series]:
    """
    Generates data for shifts.
    :param shift: int
    :return:
    """
    n = 20
    data = []
    past = []
    future = []
    for i in range(1, n + 1):
        data.append(float(i))
        if i <= shift + 1:
            past.append(float(1))
        else:
            past.append(float(i - shift))
        if i < n - shift:
            future.append(float(i + shift))
        else:
            future.append(float(n))
    data = DataFrame(data, columns=["DATA"])
    past = Series(past)
    future = Series(future)

    return data, past, future


def _do_transformation(type_of_method: str, df: DataFrame, attrs: List[str], grouping_fun: str, pom: int = 0) -> Series:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param df: DataFrame. Dataframe to be grouped.
    :param attrs: List[str]. List of parameters to be grouped. Datetime has to be included as first one.
    :param grouping_fun: str. Function to be performed.
    :param pom. Any. Additional parameter if needed.
    :return: Series
    """
    transformer = GroupDFColumnsTransformer()
    if type_of_method == FP:
        series_out = transformer.fit_predict(df, attrs, grouping_fun, pom)
    elif type_of_method == P:
        transformer.fit(df, attrs, grouping_fun, pom)
        series_out = transformer.predict()
    else:
        raise NoProperOptionInIf(transformer.get_class_info().class_name)
    return series_out


@pytest.mark.parametrize("type_of_method, df, attrs, grouping_fun, correct_list",
                         [
                             (FP, DF_1, ATTRS_1, "mean", CORRECT_DF_1_MEAN),
                             (P, DF_1, ATTRS_1, "mean", CORRECT_DF_1_MEAN),
                             (FP, DF_1, ATTRS_1, "std", CORRECT_DF_1_STD),
                             (P, DF_1, ATTRS_1, "std", CORRECT_DF_1_STD),
                             (FP, DF_1, ATTRS_1, "sum", CORRECT_DF_1_SUM),
                             (P, DF_1, ATTRS_1, "sum", CORRECT_DF_1_SUM),
                             (FP, DF_1, ATTRS_2, "sum", CORRECT_DF_2_SUM),
                             (P, DF_1, ATTRS_2, "sum", CORRECT_DF_2_SUM),
                             (FP, DF_1, ATTRS_2, "empty", CORRECT_NAN),
                             (P, DF_1, ATTRS_2, "empty", CORRECT_NAN),
                             (FP, DF_1, ATTR_MINUS, "minus", CORRECT_MINUS),
                             (P, DF_1, ATTR_MINUS, "minus", CORRECT_MINUS),
                         ])
def test_transformation(type_of_method: str, df: DataFrame, attrs: List[str], grouping_fun: str,
                        correct_list: List[float]) -> None:
    """
    Tests the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param df: DataFrame. Dataframe to be grouped.
    :param attrs: List[str]. List of parameters to be grouped. Datetime has to be included as first one.
    :param correct_list: List[float]. List of correct values.
    """
    out = _do_transformation(type_of_method, df, attrs, grouping_fun)
    assert list(out) == correct_list


@pytest.mark.parametrize("type_of_method, shift",
                         [
                             (FP, 1),
                             (P, 1),
                             (FP, 2),
                             (P, 2),
                             (FP, 3),
                             (P, 3)
                         ])
def test_shifts(type_of_method: str, shift: int) -> None:
    """
    Tests the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param shift: int. Shift to be applied.
    """
    data, past, future = generate_list_data_past_and_future(shift)
    out_past = _do_transformation(type_of_method, data, ["DATA"], "shift_past", shift)
    out_future = _do_transformation(type_of_method, data, ["DATA"], "shift_future", shift)
    print(past)
    print(out_past)
    assert past.equals(out_past) and future.equals(out_future)


@pytest.mark.parametrize("type_of_method",
                         [
                             (FP),
                             (P)
                         ])
def test_minus_with_multi_column(type_of_method: str) -> None:
    """
    Tests None output for minus with multicolumn.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    """
    out = _do_transformation(type_of_method, DF_1, ATTRS_2, "minus", 0)
    assert out is None


def test_incorrect_input() -> None:
    """
    Tests exception raise situation when there is no option selected.
    """
    env = Envs()
    env.set_running_unit_tests()
    transformer = GroupDFColumnsTransformer()
    with pytest.raises(NoProperOptionInIf):
        transformer.fit(DF_1, ATTRS_1, "wrong_function_name")


def test_none_inverse() -> None:
    """
    Test none in inverse
    """
    transformer = GroupDFColumnsTransformer()
    assert transformer.inverse() is None