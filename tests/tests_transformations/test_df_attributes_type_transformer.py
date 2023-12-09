"""
Tester
"""

from datetime import datetime
from typing import Dict

import pytest
from pandas import DataFrame

from src.constants.global_constants import FP, P
from src.data.df_explorer import DFExplorer
from src.exceptions.development_exception import NoProperOptionInIf
from src.transformations.df_attributes_type_transformer import DFAttributesTypeTransformer

ATTR_AND_DTYPES = {"FLOAT": "float64", "STR": "object", "INT": "int64", "CATEGORICAL": "category",
                   "DATETIME": "datetime64[ns]", "OBJECT": "object"}


def _create_data_frame() -> DataFrame:
    data = [[2.3, "2.4", 10.4, 1, "8"], ["Hello", "Bye", "Hi", "Ola", "Ahoj"], [1.0, 3, "1", "2", 10],
            ["cat_1", "cat_2", "cat_2", "cat_1", "cat_1"],
            [datetime(2020, 4, 5), datetime(2020, 4, 6), datetime(2020, 4, 7), datetime(2020, 4, 8),
             datetime(2020, 4, 9)],
            ["Object", "Object", "Object", "Object", "Object"]
            ]
    data = list(map(list, zip(*data)))  # type:ignore

    df = DataFrame(data=data, columns=["FLOAT", "STR", "INT", "CATEGORICAL", "DATETIME", "OBJECT"])

    return df


def _transform(type_of_method: str, df: DataFrame, attr_and_dtypes: Dict[str, str]) -> DataFrame:
    transformer = DFAttributesTypeTransformer()
    df_transformed: Dict[str, str]
    if type_of_method == FP:
        df_transformed = transformer.fit_predict(df, attr_and_dtypes)
    elif type_of_method == P:
        df_transformed = transformer.predict(df, attr_and_dtypes)
    else:
        raise NoProperOptionInIf(transformer.get_class_info().class_name)

    return df_transformed


@pytest.mark.parametrize("type_of_method, input_df, attr_and_dtypes",
                         [(FP, _create_data_frame(), ATTR_AND_DTYPES),
                          (P, _create_data_frame(), ATTR_AND_DTYPES)]
                         )
def test_types_dataframe(type_of_method: str, input_df: DataFrame, attr_and_dtypes: Dict[str, str]) -> None:
    """
    Tests transformation for DataFrame.
    :param type_of_method: str. P for "p" (predict) and FP for "fp" (fit_predict).
    :param input_df: DataFrame. Pandas structure to have its value attribute transformed.
    :param attr_and_dtypes: Dict[str, str]. Correct dtypes.
    """
    dfe = DFExplorer()

    df_output = _transform(type_of_method, input_df, attr_and_dtypes)
    assert dfe.get_df_types(df_output) == attr_and_dtypes


def test_empty_fit() -> None:
    """
    Tests empty output of fit.
    """
    transformer = DFAttributesTypeTransformer()
    assert transformer.fit(_create_data_frame(), ATTR_AND_DTYPES) is None


def test_none_inverse() -> None:
    """
    Test none in inverse
    """
    transformer = DFAttributesTypeTransformer()
    assert transformer.inverse() is None
