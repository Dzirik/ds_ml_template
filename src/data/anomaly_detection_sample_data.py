"""
Sample data for anomaly detection.
"""
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

from numpy.random import normal, uniform, seed
from pandas import DataFrame

from src.data.splitter import Splitter
from src.visualisations.plotly_time_series import PlotlyTimeSeries

ATTR_DATE_TIME = "DATE_TIME"
ATTR_ID = "ID"
ATTRS = ["ATTR_1", "ATTR_2", "ATTR_3", "ATTR_4", "ATTR_5"]


class AnomalyDetectionSampleData:
    """
    Class for generating simple anomaly detection data.
    """

    def __init__(self, abs_values: bool = True, add_zeros: bool = False) -> None:
        """
        :param abs_values: bool. If generated values are absolute.
        :param add_zeros: bool. If to add zeros in the beginning - because of logarithic transformation.
        """
        self._abs_values = abs_values
        self._add_zeros = add_zeros
        self._first_date = datetime(year=1980, month=1, day=1)
        self._df = DataFrame()
        self._params: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = [
            ((0., 1.), (2., 2.), 0.10),
            ((1., 3.), (2., 5.), 0.10),
            ((0., 1.), (3., 10), 0.025),
            ((1., 3.), (6., 12.), 0.025),
            ((5., 1.), (5., 15.), 0.05)
        ]

    def _generate_date_time(self, n: int) -> List[datetime]:
        """
        Generates the date time attributes.
        :param n: int. Number of observation.
        :return: List[datetime].
        """
        return [self._first_date + timedelta(days=x) for x in range(n)]

    @staticmethod
    def _generate_id(n: int) -> List[int]:
        """
        Generates id data column.
        :param n: int. Number of observations.
        :return: List[int].
        """
        return list(range(n))

    def _generate_one_attribute(self, n: int, distribution_main: Tuple[float, float], distribution_contamination: \
            Tuple[float, float], probability_contamination: float) -> List[float]:
        """
        Generates random data based on parameters.
        :param n: int. Number of observation.
        :param distribution_main: Tuple[float, float]. Main distribution parameters (<mean>, <std>) for its normal
                                  distribution.
        :param distribution_contamination: Tuple[float, float]. Contamination distribution parameters (<mean>, <std>)
                                  for its normal distribution.
        :param probability_contamination: float. Probability of raising observation from contamination distribution.
        :return: List[float]. List of numbers.
        """
        add = []
        if self._add_zeros:
            add = [0.]
            n = n - 1
        if self._abs_values:
            return add + [abs(normal(distribution_main[0], distribution_main[1])) if uniform() >= \
                                                                                     probability_contamination \
                              else abs(normal(distribution_contamination[0], distribution_contamination[1])) \
                          for _ in range(n)]
        return add + [normal(distribution_main[0], distribution_main[1]) if uniform() >= probability_contamination \
                          else \
                          normal(distribution_contamination[0], distribution_contamination[1]) for _ in range(n)]

    def generate(self, n: int, seed_number: Optional[int] = None) -> DataFrame:
        """
        Generates the data.
        :param n: int. Number of observation.
        :param seed_number: Optional[int].
        :return: DataFrame.
        """
        self._df = DataFrame()
        self._df[ATTR_DATE_TIME] = self._generate_date_time(n)
        self._df[ATTR_ID] = self._generate_id(n)
        self._df.index = self._df[ATTR_DATE_TIME]
        self._df.index.name = ""
        seed(seed_number)
        for attr, params in zip(ATTRS, self._params):
            d_main, d_cont, p = params
            self._df[attr] = self._generate_one_attribute(n, d_main, d_cont, p)
        return self._df

    def create_list_of_data_frames(self, n: int, seed_number: Optional[int] = None) -> Tuple[
        List[DataFrame], DataFrame]:
        """
        Creates sample data of four separated data frames and concatenated one.
        :param n: int. Number of observation.
        :param seed_number: Optional[int].
        :return: Tuple[List[DataFrame], DataFrame]. (<List of data frames>, <whole data frame>).
        """
        splitter = Splitter()
        df_whole = self.generate(n=n, seed_number=seed_number)
        attrs = list(df_whole.columns)
        df_1, df_pom = splitter.create_train_test_split_time_series_df(df_whole, attrs, 0.25)
        df_2, df_pom = splitter.create_train_test_split_time_series_df(df_pom, attrs, 1 / 3)
        df_3, df_4 = splitter.create_train_test_split_time_series_df(df_pom, attrs, 0.5)

        return [df_1, df_2, df_3, df_4], df_whole


def plot_data_frame_series(df: DataFrame, attrs: List[str], plot_title: str = "Time Series Data") -> None:
    """
    Plots the attributes from the data frames with date time index - as a time series.
    :param df: DataFrame.
    :param attrs: List[str]. List of attributes to be plotted.
    :param plot_title: str.
    """
    ts_visu = PlotlyTimeSeries()

    series = []
    for attr in attrs:
        series.append(df[attr])

    ts_visu.set_selectors(selectors=[
        {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
        {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
        {"step": "all"}
    ])

    ts_visu.plot(
        series=series,
        series_names=attrs,
        plot_title=plot_title,
        y_title="Value"
    )
