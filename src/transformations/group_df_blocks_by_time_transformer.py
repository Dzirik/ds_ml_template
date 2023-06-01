"""
Transformer

Note: The implementation is naive with respect to complexity. But fits the current need. In future
      it can be used to to test the proper possibility.

flooring: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.floor.html
Aliases: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
- "1d" - days
- "12h" - hours
- "4min" - mins
"""
from pandas import DataFrame, concat

from src.exceptions.development_exception import NoProperOptionInIf
from src.exceptions.exception_executioner import ExceptionExecutioner
from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class GroupDFBlocksByTimeTransformer(BaseTransformer):  # type:ignore
    """
    Takes a dataframe with one time column (datetime) and some other columns. Groups the data frame based on
    windowed time column and within this window groups all the other columns with grouping function.

    Note: The implementation is naive with respect to complexity. But fits the current need. In future
          it can be used to to test the proper possibility.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(input_type=[DataFrame], input_elements_type=[None],
                                                         output_type=[DataFrame], output_elements_type=[None])
        BaseTransformer.__init__(self, class_name="GroupDFBlocksByTimeTransformer",
                                 transformer_description=transformer_description)

        self._df_out: DataFrame

    # pylint: disable=arguments-differ
    # pylint: disable=arguments-renamed
    def fit(self, df: DataFrame, attr_time: str = "DATETIME", window_length: str = "1d",
            grouping_fun: str = "mean") -> None:
        """
        Fits.
        :param df: DataFrame. Dataframe to be grouped.
        :param attr_time: str. Name of the attribute to be grouped by.
        :param window_length: str. Window length string format from here
               https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.floor.html
        :param grouping_fun: str. Function to be performed. Currently available "mean", "max", "min", "first", "last".
        """
        df_new = df.copy()

        # grouping time attribute shirking
        df_new[attr_time] = df_new[attr_time].dt.floor(window_length)

        # creating groups for grouping based on new time attribute
        attrs = list(df_new.columns)
        attrs.remove(attr_time)
        grp = df_new.groupby(by=attr_time)[attrs]

        # output df
        self._df_out = DataFrame([], columns=[attr_time, grouping_fun.upper()])

        # grouping blocks - this implementation is naive, but based on need sufficient
        for key, _ in grp:
            # group.stack().std()
            # group.values.std(ddof=0)
            # grp.get_group(key)
            if grouping_fun == "mean":
                value = grp.get_group(key).values.mean()
            elif grouping_fun == "std":
                value = grp.get_group(key).values.std()
            elif grouping_fun == "min":
                value = grp.get_group(key).values.min()
            elif grouping_fun == "max":
                value = grp.get_group(key).values.max()
            elif grouping_fun == "first":
                value = grp.get_group(key).values[0][0]
            elif grouping_fun == "last":
                value = grp.get_group(key).values[-1][0]
            elif grouping_fun == "sum":
                value = grp.get_group(key).values.sum()
            else:
                ExceptionExecutioner(NoProperOptionInIf).log_and_raise(description=self._class_info.class_type + " " +
                                                                                   self._class_info.class_name)
            df_pom = DataFrame([[key, value]], columns=[attr_time, grouping_fun.upper()])
            self._df_out = concat([self._df_out, df_pom])

        # sorting
        self._df_out.sort_values(by=attr_time, inplace=True, ascending=True)
        self._df_out.reset_index(drop=True, inplace=True)

    def fit_predict(self, df: DataFrame, attr_time: str = "DATETIME", window_length: str = "1d",
                    grouping_fun: str = "mean") -> DataFrame:
        """
        Fits and predicts.
        :param df: DataFrame. Dataframe to be grouped.
        :param attr_time: str. Name of the attribute to be grouped by.
        :param window_length: str. Window length string format from here
               https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.floor.html
        :param grouping_fun: str. Function to be performed. Currently available "mean", "max", "min".
        :return: DataFrame. Grouped data frame.
        """
        self.fit(df, attr_time, window_length, grouping_fun)
        return self._df_out

    def predict(self) -> DataFrame:
        """
        Predicts
        :return: DataFrame. Dataframe created in fit/fit_predict.
        """
        return self._df_out

    def inverse(self) -> None:
        """
        Does the inverse transformation.
        """
        return None
    # pylint: enable=arguments-differ
    # pylint: enable=arguments-renamed
