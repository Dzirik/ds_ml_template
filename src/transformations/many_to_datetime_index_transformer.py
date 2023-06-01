"""
Transformer
"""
from datetime import datetime
from typing import List, Union

from pandas import DatetimeIndex, Series, DataFrame

from src.exceptions.data_exception import IncorrectDataStructure
from src.exceptions.development_exception import NoProperOptionInIf
from src.exceptions.exception_executioner import ExceptionExecutioner
from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class ManyToDatetimeIndexTransformer(BaseTransformer):  # type:ignore
    """
    Transforms Series, DataFrame, List of datetime values to DatetimeIndex.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(input_type=[Series, DataFrame, List],
                                                         input_elements_type=[datetime], output_type=[DatetimeIndex],
                                                         output_elements_type=[None])
        BaseTransformer.__init__(self, class_name="ManyToDatetimeIndexTransformer",
                                 transformer_description=transformer_description)

    def _transform(self, data: Union[DataFrame, Series, List[datetime]]) -> DatetimeIndex:
        """
        Transforms a dataframe of one column, time series or list of datetimes to DatetimeIndex.
        :param data: Union[DataFrame, Series, List[datetime]]. Object to be transformed.
        :return: DatetimeIndex.
        """
        if isinstance(data, DataFrame):
            if data.shape[1] == 1:
                attr_name = data.columns[0]
                output = DatetimeIndex(data[attr_name])
            else:
                text = ".DataFrame does not consist of only one column."
                ExceptionExecutioner(IncorrectDataStructure).log_and_raise(
                    description=self._class_info.class_type + " " + self._class_info.class_name + text)
        elif isinstance(data, Series):
            output = data.index
        elif isinstance(data, list):
            output = DatetimeIndex(data)
        else:
            ExceptionExecutioner(NoProperOptionInIf).log_and_raise(description=self._class_info.class_type + " " +
                                                                               self._class_info.class_name)
        return output

    # pylint: disable=arguments-differ
    def fit(self, data: Union[DataFrame, Series, List[datetime]]) -> None:
        pass

    def fit_predict(self, data: Union[DataFrame, Series, List[datetime]]) -> DatetimeIndex:
        return self._transform(data)

    def predict(self, data: Union[DataFrame, Series, List[datetime]]) -> DatetimeIndex:
        return self._transform(data)

    def inverse(self) -> None:
        """
        Does the inverse transformation.
        """
        return None
    # pylint: enable=arguments-differ
