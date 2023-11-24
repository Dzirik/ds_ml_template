"""
Transformer

Notes:
- It is very simple, but I need it for pipelines.
- In general for future extension can be used:
    - df.applymap(SomeFunction)
    - df[list_of_columns].apply(SomeFunction)
"""
from typing import List, Any, Dict

from pandas import DataFrame, Series

from src.exceptions.development_exception import NoProperOptionInIf
from src.exceptions.exception_executioner import ExceptionExecutioner
from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class GroupDFColumnsTransformer(BaseTransformer):  # type:ignore
    """
    Takes a dataframe with several columns and performs some operation on those columns - groups columns in one row to
    get some output.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(input_type=[DataFrame], input_elements_type=[None],
                                                         output_type=[Series], output_elements_type=[None])
        BaseTransformer.__init__(self, class_name="GroupDFColumnsTransformer",
                                 transformer_description=transformer_description)

        self._series_out: Series

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-branches
    def fit(self, df: DataFrame, attrs: List[str], grouping_fun: str, params: Dict[str, Any]) -> None:
        """
        Fits.
        - "shift_*": params = {"shift": int}.
        - "div": params = {"zero_div_replacement": float}
        :param df: DataFrame. Dataframe to be grouped.
        :param attrs: List[str]. List of parameters to be grouped. Datetime has to be included as first one.
        :param grouping_fun: str. Function to be performed. Currently available "mean", "std", "sum", "empty"
                                  "shift_future", "shift_past", "minus", "div".
        :param params. Dict[str, Any]. Additional parameters if needed.
        """
        if grouping_fun == "mean":
            self._series_out = df[attrs].mean(axis=1)
        elif grouping_fun == "std":
            self._series_out = df[attrs].std(axis=1)
        elif grouping_fun == "sum":
            self._series_out = df[attrs].sum(axis=1)
        elif grouping_fun == "empty":
            self._series_out = Series(data=[""] * df.shape[0], index=df.index)
        elif grouping_fun == "shift_future":
            pom = params["shift"]
            self._series_out = df[attrs[0]].tolist()
            self._series_out = Series(self._series_out[pom:] + [self._series_out[-1]] * pom, index=df.index)
        elif grouping_fun == "shift_past":
            pom = params["shift"]
            self._series_out = df[attrs[0]].tolist()
            self._series_out = Series([self._series_out[0]] * pom + self._series_out[:-pom], index=df.index)
        elif grouping_fun == "minus":
            if len(attrs) == 1:
                self._series_out = - df[attrs[0]]
            else:
                self._series_out = None
        elif grouping_fun == "div":
            if len(attrs) == 2:
                pom = params["zero_div_replacement"]
                numerator = list(df[attrs[0]])
                denominator = list(df[attrs[1]])
                result = [x / y if y != 0 else pom for x, y in zip(numerator, denominator)]
                self._series_out = Series(result, index=df.index)
            else:
                self._series_out = None
        else:
            ExceptionExecutioner(NoProperOptionInIf).log_and_raise(description=self._class_info.class_type + " " +
                                                                               self._class_info.class_name)

    # pylint: enable=too-many-branches

    def fit_predict(self, df: DataFrame, attrs: List[str], grouping_fun: str, params: Dict[str, Any]) -> Series:
        """
        Fits and predicts.
        :param df: DataFrame. Dataframe to be grouped.
        :param attrs: List[str]. List of parameters to be grouped. Datetime has to be included as first one.
        :param grouping_fun: str. Function to be performed. Currently available.
        :param params. Dict[str, Any]. Additional parameters if needed.
        :return: Series. Grouped data series.
        """
        self.fit(df, attrs, grouping_fun, params)
        return self._series_out

    def predict(self) -> Series:
        """
        Predicts
        :return: Series. Series created in fit/fit_predict.
        """
        return self._series_out

    def inverse(self) -> None:
        """
        Does the inverse transformation.
        """
        return None
    # pylint: enable=arguments-differ
