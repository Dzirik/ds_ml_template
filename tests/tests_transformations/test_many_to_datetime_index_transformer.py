"""
Tester

If using an exception assert statement, use following:
    env = Envs()
    env.set_running_unit_tests()
"""
from datetime import datetime
from typing import List, Union

import pytest
from pandas import DatetimeIndex, DataFrame, Series

from src.constants.global_constants import FP, P
from src.exceptions.data_exception import IncorrectDataStructure
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.many_to_datetime_index_transformer import ManyToDatetimeIndexTransformer
from src.utils.envs import Envs

TS_DF = DataFrame({"TIME": [datetime(2018, 2, 5), datetime(2018, 3, 11), datetime(2018, 6, 6), datetime(2018, 7, 1),
                            datetime(2018, 7, 29), datetime(2019, 1, 3), datetime(2019, 5, 19), datetime(2019, 7, 4),
                            datetime(2019, 7, 11), datetime(2019, 12, 4)]})
TS_SERIES = Series(index=TS_DF["TIME"].to_list(), data=[3, 72, 1, 41, 77, 99, 62, 5, 7, 6])
TS_LIST = TS_DF["TIME"].to_list()

OUTPUT = TS_SERIES.index


def _convert_do_datetime_index(type_of_method: str, data: Union[DataFrame, Series, List[datetime]]) -> DatetimeIndex:
    transformer = ManyToDatetimeIndexTransformer()
    output: DatetimeIndex

    if type_of_method == FP:
        output = transformer.fit_predict(data)
    elif type_of_method == P:
        output = transformer.predict(data)
    else:
        raise NoProperOptionInIf(transformer.get_class_info().class_name)
    return output


@pytest.mark.parametrize("type_of_method, data, correct_output",
                         [(P, TS_DF, OUTPUT),
                          (FP, TS_DF, OUTPUT),
                          (P, TS_SERIES, OUTPUT),
                          (FP, TS_SERIES, OUTPUT),
                          (P, TS_LIST, OUTPUT),
                          (FP, TS_LIST, OUTPUT)]
                         )
def test_transform(type_of_method: str, data: Union[DataFrame, Series, List[datetime]], correct_output: DatetimeIndex) \
        -> None:
    """
    Tests correctness of output.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param data: Union[DataFrame, Series, List[datetime]]. Object to be transformed.
    :param correct_output: DatetimeIndex.
    """
    output = _convert_do_datetime_index(type_of_method, data)
    assert sum(output == correct_output) == len(correct_output)


def test_no_proper_if() -> None:
    """
    Tests wrong input type.
    """
    env = Envs()
    env.set_running_unit_tests()
    transformer = ManyToDatetimeIndexTransformer()
    with pytest.raises(NoProperOptionInIf):
        transformer.fit_predict(10)


def test_wrong_dim_of_data_frame() -> None:
    """
    Tests wrong data frame.
    """
    env = Envs()
    env.set_running_unit_tests()
    incorrect_data_frame = DataFrame([[datetime(2018, 2, 5), datetime(2018, 3, 11)],
                                      [datetime(2018, 2, 5), datetime(2018, 3, 11)]])
    transformer = ManyToDatetimeIndexTransformer()
    with pytest.raises(IncorrectDataStructure):
        transformer.fit_predict(incorrect_data_frame)


def test_empty_fit() -> None:
    """
    Tests empty output of fit.
    """
    transformer = ManyToDatetimeIndexTransformer()
    assert transformer.fit(TS_DF) is None


def test_none_inverse() -> None:
    """
    Test none in inverse
    """
    transformer = ManyToDatetimeIndexTransformer()
    assert transformer.inverse() is None
