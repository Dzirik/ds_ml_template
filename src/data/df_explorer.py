"""
Simple class for displaying basic stats about data frame in exploration stage.
"""

from typing import Dict, List

from pandas import DataFrame
from tabulate import tabulate

from src.visualisations.plotly_bar_chart import PlotlyBarChart
from src.visualisations.plotly_histogram import PlotlyHistogram


class DFExplorer:
    """
    Prints and plots basic information about a DataFrame.
    """

    def __init__(self) -> None:
        self._histogram = PlotlyHistogram()
        self._bar_chart = PlotlyBarChart()

    def print_info_about_data_frame(self, df: DataFrame) -> None:
        """
        Print overall df stats.
        :param df: DataFrame. Data to have its stats printed.
        """
        print(f"DataFrame type: {str(type(df))}")
        print(f"DataFrame shape: {str(df.shape)}")
        print(f"DataFrame dtypes: {self.get_df_types(df)}")
        print("\n")
        print(f"DataFrame head:\n{df.head()}")
        print("\n")
        print(f"DataFrame description:\n{df.describe()}")
        print("\n")

    @staticmethod
    def get_df_types(df: DataFrame) -> Dict[str, str]:
        """
        Gives the data type per column of a DataFrame.
        :param df: DataFrame.
        :return: Dict[str, str].
        """
        df_types: Dict[str, str] = df.dtypes.apply(lambda x: x.name).to_dict()
        return df_types

    @staticmethod
    def get_memory_usage(df: DataFrame, attr_name: str, list_dtypes: List[str], deep: bool = True) -> None:
        """
        Returns the memory usage of an attribute of a DataFrame upon the type of dtype of the values.
        :param df: DataFrame.
        :param attr_name: str. Name of the attribute to which the measurement is done.
        :param list_dtypes: List[str]. List of dtypes to be measured.
        :param deep: bool. If True, introspect the data deeply by interrogating object dtypes for system-level memory
        consumption.
        """
        print(f"Memory usage for attribute: {attr_name}")
        for data_type in list_dtypes:
            print(f"  Attribute Name: {attr_name}")
            print(f"  Measured dtype: {data_type}")
            try:
                print("  Memory Usage:", df[attr_name].astype(data_type).memory_usage(deep=deep))
            # pylint: disable=broad-except
            except Exception as exc:
                print(exc)
            # pylint: enable=broad-except
            if data_type != list_dtypes[-1]:
                print("\n")

    @staticmethod
    def get_nan_stats(df: DataFrame, fraction: bool = False) -> None:
        """
        Prints the number of NaN values in the DataFrame and per attribute.
        :param df: DataFrame.
        :param fraction: bool. If True, the % of NaN values is shown with respect to the total number of rows.
        """
        print("DataFrame shape:", df.shape)
        print("Total number of NaN values:", df.isna().sum().sum())
        print("NaN values per Attribute:", "\n")
        if fraction:
            tabs = [df.columns.tolist(), df.isna().sum().tolist(), (df.isna().sum() / df.shape[0]).round(2).tolist()]
            total = ["TOTAL", df.isna().sum().sum(), (df.isna().sum() / df.shape[0]).round(2).sum()]
            for i, tab in enumerate(tabs):
                tab.append(total[i])
            print(tabulate(zip(*tabs), headers=["Attribute", "NaN values", "Fraction"]))
        else:
            print(df.isna().sum())

    @staticmethod
    def print_single_attr_stats_without_plot(df: DataFrame, attr_name: str, min_unique: int = 20) -> None:
        """
        Prints and plots individual stats of the df attributes.
        :param df: DataFrame. Data to have its attribute stats printed.
        :param attr_name: str. Name of the attribute to be analyzed.
        :param min_unique: int. Used to set the maximum allowed unique values for individual columns to be printed and
        plotted.
        """
        print(f"Attribute Name: {attr_name}")
        print(f" Attribute type: {df[attr_name].dtype}")
        print(f" Number of Null values: {df[[attr_name]].isnull().sum()[0]}")
        print(f" Number of unique values is:{len(df[attr_name].value_counts())}")
        print(f" Percentage of unique values is: {len(df[attr_name].value_counts()) / df.shape[0]}")
        if len(df[attr_name].value_counts()) < min_unique:
            pom = df[attr_name].value_counts()
            print("\n")
            print("Summation of unique values per ID:")
            print(pom)
        print("\n")

    def print_single_attr_stats_with_plots(self, df: DataFrame, attr_name: str, min_unique: int = 20) -> None:
        """
        Prints and plots individual stats of the df attributes.
        :param df: DataFrame. Data to have its attribute stats printed.
        :param attr_name: str. Name of the attribute to be analyzed.
        :param min_unique: int. Used to set the maximum allowed unique values for individual columns to be printed and
        plotted.
        """
        self.print_single_attr_stats_without_plot(df=df, attr_name=attr_name)
        if len(df[attr_name].value_counts()) < min_unique:
            counts = df[[attr_name]].value_counts()
            self._bar_chart.plot(
                array_ids=[cat[0] for cat in counts.index.tolist()],
                array_values=counts.values,
                plot_title="Bar Chart",
                name_ids="Count",
                name_values=attr_name
            )
        elif (df[attr_name].dtype in ("float64", "int64")) and \
                df[[attr_name]].isnull().sum()[0] == 0:
            self._histogram.plot(data=df[attr_name].to_numpy(), plot_title="Histogram", x_title=attr_name)

    def print_attr_stats(self, df: DataFrame, min_unique: int = 20) -> None:
        """
        Prints df columns stats.
        :param df: DataFrame. Data to have its column stats printed.
        :param min_unique: int. Used to set the maximum allowed unique values for individual columns to be printed and
        plotted.
        """
        for attr in df.columns:
            self.print_single_attr_stats_with_plots(df, attr, min_unique)
            print("\n")
            print("#############################################")

    @staticmethod
    def get_rows_with_any_nan(df: DataFrame) -> DataFrame:
        """
        Returns data frame with those rows from original data frame containing at least one NaN value.
        :param df: DataFrame.
        :return: DataFrame.
        """
        return df[df.isna().any(axis=1)]

    @staticmethod
    def _print_comparison(sum_1: float, sum_2: float) -> None:
        """
        Prints comparsion of two sums.
        :param sum_1: float.
        :param sum_2: float.
        """
        print(f" - Sum of First List is: {sum_1}")
        print(f" - Sum of Second List is: {sum_2}")
        print(f" - Subtraction of Sums of Lists is: {sum_1 - sum_2}")
        if sum_1 != 0:
            print(f" - Percentage of Difference (1-2)/1 is: {(sum_1 - sum_2) / sum_1} ")
        if sum_2 != 0:
            print(f" - Percentage of Difference (1-2)/2 is: {(sum_1 - sum_2) / sum_2} ")

    def compare_attributes_in_data_frames(self, df_1: DataFrame, df_2: DataFrame, attr_names_to_compare: List[str]) \
            -> None:
        """
        Computes compare_lists for attributes in two data frames.
        :param df_1: DataFrame. First data frame.
        :param df_2: DataFrame. Second data frame.
        :param attr_names_to_compare: List[str]. List of attributes to be compared. All attributes have to be in both.
        """
        print(f"Are DFs equal in pandas? {df_1.equals(df_2)}")
        print("Checking Overall Sums for DFs")
        self._print_comparison(sum(df_1.sum(axis=0)), sum(df_2.sum(axis=0)))
        for attr in attr_names_to_compare:
            print(f"Checking for Attribute: {attr}")
            self._print_comparison(sum(list(df_1[attr])), sum(list(df_2[attr])))
