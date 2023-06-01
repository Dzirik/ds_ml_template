"""
Sample data for testing operations with time series data frame.
"""
from datetime import datetime, timedelta
from typing import List

from pandas import DataFrame

from src.transformations.many_to_datetime_index_transformer import ManyToDatetimeIndexTransformer


class TimeSeriesOneMinuteData:
    """
    One minute data for testing time series data frame operations.
    """
    _ATTRS_DATA = ["1", "2", "I", "-I", "2*I", "5*I", "10*I"]
    _ATTRS = ["DATETIME"] + _ATTRS_DATA

    def __init__(self, n: int = 44640) -> None:
        """
        :param n: int. Size of data frame. Default is one month (January 2000, 44 600mins).
        """
        self._n = n
        self._df = DataFrame()

    def get_attrs(self) -> List[str]:
        """
        Gets the attribute names.
        :return: List[str].
        """
        return self._ATTRS

    def get_data_frame(self) -> DataFrame:
        """
        Gets the sample data frame.
        :return: DataFrame.
        """
        date_time = []
        data = []
        first_date = datetime(2000, 1, 1, 0, 0, 0)

        for i in range(self._n):
            date_time.append(first_date + timedelta(minutes=i))
            data.append([1, 2, i, -i, 2 * i, 5 * i, 10 * i])
        date_time_index = ManyToDatetimeIndexTransformer().fit_predict(date_time)

        self._df = DataFrame(data=data, columns=self._ATTRS_DATA)
        self._df["DATETIME"] = date_time_index
        self._df = self._df[self._ATTRS]
        self._df.index = date_time_index

        return self._df
