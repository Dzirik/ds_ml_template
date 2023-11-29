"""
Transformer

Sets proper type for every attribute in a dataframe.
"""

from typing import Dict

from pandas import DataFrame

from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class DFAttributesTypeTransformer(BaseTransformer):  # type:ignore
    """
    Sets a given type for every attribute of a dataframe based on a dictionary. Please see a doctest for more
    information.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(input_type=[DataFrame], input_elements_type=[None],
                                                         output_type=[DataFrame], output_elements_type=[None])
        BaseTransformer.__init__(self, class_name="DFAttributesTypeTransformer",
                                 transformer_description=transformer_description)

    @staticmethod
    def _transform(df: DataFrame, attr_and_dtypes: Dict[str, str]) -> DataFrame:
        """
        Sets a given type for every attribute of a dataframe based on a dictionary.
        :param df: DataFrame.
        :param attr_and_type: Dict[str, str]. Dictionary of columns and respective dtypes. Please see doctest for more
        information.
        :return: DataFrame.
        """
        for attr in df:
            df[attr] = df[attr].astype(attr_and_dtypes[attr])

        return df

    # pylint: disable=arguments-differ
    def fit(self, df: DataFrame, attr_and_dtypes: Dict[str, str]) -> None:
        pass

    def fit_predict(self, df: DataFrame, attr_and_dtypes: Dict[str, str]) -> DataFrame:
        return self._transform(df, attr_and_dtypes)

    def predict(self, df: DataFrame, attr_and_dtypes: Dict[str, str]) -> DataFrame:
        return self._transform(df, attr_and_dtypes)

    def inverse(self) -> None:
        """
        Does the inverse transformation.
        """
        return None
    # pylint: enable=arguments-differ
