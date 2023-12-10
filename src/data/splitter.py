"""
Splitter

There is a documentation notebook notebooks/documentation/models_assessment_01_basic_splits.py.
"""

from datetime import datetime, timedelta
from typing import Tuple, Optional, Any, List, Union

from numpy import ndarray, dtype, array
from pandas import DataFrame, Series, concat
# pylint: disable=no-name-in-module
from pandas._libs.tslibs.timestamps import Timestamp
# pylint: enable=no-name-in-module
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split


class Splitter:
    """
    Splits Data into training and testing set for classic X, Y data set and time series data.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def create_train_test_split_np(X: ndarray[Any, dtype[Any]], Y: ndarray[Any, dtype[Any]], cv_train_size: float,
                                   cv_random_state: Optional[int] = None, shuffle: bool = True,
                                   stratify: Optional[Any] = None) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]],
                  ndarray[Any, dtype[Any]]]:
        """
        Splits X, Y into training and testing sets.
        :param X: ndarray[Any, dtype[Any]]. Input variable.
        :param Y: ndarray[Any, dtype[Any]]. Output variable - SHOULD BE TWO DIMENSION
        :param cv_train_size: float. Size of the training set - between 0,1
        :param cv_random_state: Optional[int]. Random number for splitting.
        :param shuffle: bool. If to shuffle.
        :param stratify: Optional[Any]. If to balance the classes.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]],
                ndarray[Any, dtype[Any]]]. X_train, X_test, Y_train, Y_test.
        """
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=cv_train_size, random_state=cv_random_state, shuffle=shuffle, stratify=stratify

        )
        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def create_equal_time_period_split(start_time: datetime, end_time: datetime, n_periods: int, \
                                       only_inside_splits: bool = False) -> List[datetime]:
        """
        Splits time period into equidistant intervals and returns those intervals boundaries.
        :param start_time: datetime.
        :param end_time: datetime.
        :param n_periods: int. Number of periods to be created.
        :param only_inside_splits: bool. If to exclude beginning and the end.
        :return: List[datetime].
        """
        if n_periods < 2:
            return []
        delta = (end_time - start_time) / n_periods
        if only_inside_splits:
            return [start_time + delta * i for i in range(1, n_periods)]
        return [start_time + delta * i for i in range(0, n_periods + 1)]

    def create_train_test_split_df(self, df: DataFrame, attrs_x: List[str], attrs_y: List[str],
                                   attr_date_time: Optional[str], \
                                   cv_random_state: Optional[int] = None, cv_train_size: float = 0.75) -> \
            Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Classic split of a data frame. If attr_date_time is given, it creates a datetime index out of it. Please
        see the documentation notebook for more details.
        Used for AD.s
        :param df: DataFrame. DF to be used for cv.
        :param attrs_x: List[str].
        :param attrs_y: List[str].
        :param attr_date_time: Optional[str].
        :param cv_random_state: Optional[int].
        :param cv_train_size: float.
        :return: Tuple[DataFrame, DataFrame, DataFrame, DataFrame]. df_train_X, df_test_X, df_train_Y, df_test_Y.
        """
        X_train, X_test, Y_train, Y_test = self.create_train_test_split_np(
            X=df[attrs_x].to_numpy(),
            Y=df[attrs_y].to_numpy(),
            cv_train_size=cv_train_size,
            cv_random_state=cv_random_state,
            shuffle=True
        )

        df_train_X = DataFrame(X_train, columns=attrs_x)
        if attr_date_time is not None:
            df_train_X.index = df_train_X[attr_date_time]
            df_train_X.index.name = ""
        df_train_Y = DataFrame(Y_train, columns=attrs_y)

        df_test_X = DataFrame(X_test, columns=attrs_x)
        if attr_date_time is not None:
            df_test_X.index = df_test_X[attr_date_time]
            df_test_X.index.name = ""
        df_test_Y = DataFrame(Y_test, columns=attrs_y)

        return df_train_X, df_test_X, df_train_Y, df_test_Y

    @staticmethod
    def create_train_test_split_time_series_series(ts: Series, cv_train_size: float) -> Tuple[Series, Series]:
        """
        Splits time series, actually gets first cv_train_size and the rest, no randomness.
        :param ts: Series. Time series.
        :param cv_train_size: Union[None, float]. Size of the training set - between 0,1.
        :return: Tuple[Series, Series]. ts_train, ts_test.
        """
        ts_train, ts_test = ts[:int(len(ts) * cv_train_size)], ts[int(len(ts) * cv_train_size):]

        return ts_train, ts_test

    def create_train_test_split_time_series_df(self, df: DataFrame, attrs: List[str], cv_train_size: float = 0.75) -> \
            Tuple[DataFrame, DataFrame]:
        """
        Splits data frame with respect to splitters time series cv split.
        Used for AD.
        :param df: DataFrame.
        :param attrs: List[str]. List of attributes to be split.
        :param cv_train_size: float.
        :return: Tuple[DataFrame, DataFrame]. <df_train, df_test>.
        """
        df_train = DataFrame()
        df_test = DataFrame()
        for attr in attrs:
            ts_train, ts_test = self.create_train_test_split_time_series_series(df[attr], cv_train_size=cv_train_size)
            df_train[attr] = ts_train
            df_test[attr] = ts_test
        return df_train, df_test

    def create_train_test_split_of_list_of_objects(self, list_of_items: List[Any], cv_train_size: float,
                                                   cv_random_state: Optional[int] = None) \
            -> Tuple[List[Any], List[Any]]:
        """
        Classic CV split of the items in a list.
        :param list_of_items: List[Any]. List of items to be split into train and test list.
        :param cv_train_size: float.
        :param cv_random_state: Optional[int].
        :return: Tuple[List[Any], List[Any]]. <list_train, list_test>.
        """
        ids: ndarray[Any, dtype[Any]] = array(list(range(len(list_of_items)))).reshape((len(list_of_items), 1))
        ids_train_array, ids_test_array, _, _ = self.create_train_test_split_np(
            X=ids, Y=ids, cv_train_size=cv_train_size, cv_random_state=cv_random_state, shuffle=True
        )

        ids_train = list(ids_train_array.reshape((len(ids_train_array),)))
        ids_train.sort()

        ids_test = list(ids_test_array.reshape((len(ids_test_array),)))
        ids_test.sort()

        return [list_of_items[i] for i in ids_train], [list_of_items[i] for i in ids_test]

    @staticmethod
    def create_loo_split_of_list_of_objects(list_of_items: List[Any]) -> List[Tuple[List[Any], List[Any]]]:
        """
        Creates
        :param list_of_items: List[Any]. List of items to be split.
        :return: List[Tuple[List[Any], List[Any]]]. List of tuples (<train list>, <test_list>).
        """
        loo = LeaveOneOut()
        _ = loo.get_n_splits(list_of_items)

        train_test_splits = []
        for _, (train_index, test_index) in enumerate(loo.split(list_of_items)):
            train = [list_of_items[i] for i in train_index]
            test = [list_of_items[i] for i in test_index]
            train_test_splits.append((train, test))

        return train_test_splits

    @staticmethod
    def split_data_frame_by_split_dates(df: DataFrame, attr_date_time: str, split_dates: \
            List[Union[Timestamp, datetime]]) -> List[DataFrame]:
        """
        Sorts the data frame and splits it based on split dates (only in time zone, edges are added automatically).
        Splits the data frame by those splits and returns the list of splitted data frames.
        :param df: DataFrame.
        :param attr_date_time: str. Attribute name for time attribute.
        :param split_dates: Union[Timestamp, datetime]. Dates for splits in the date_time. EDGES ARE ADDED
                                AUTOMATICALLY. This is not very good to mix format, but both work. So I keep them
                                both in tests.
        :return: List[DataFrame]. List of splits
        """
        df = df.sort_values(by=attr_date_time)

        date_time = df[attr_date_time]
        split_dates = [date_time[0]] + split_dates + [date_time[-1] + timedelta(days=1)]
        split_dates.sort(reverse=False)

        dfs = []
        for date_from, date_to in zip(split_dates[0:-1], split_dates[1:]):
            take = (date_from <= date_time) & (date_time < date_to)
            df_pom = df.loc[take,]
            df_pom = df_pom.sort_values(by=attr_date_time)
            dfs.append(df_pom)

        # check
        df_check = concat(dfs)
        df_check = df_check.sort_values(by=attr_date_time)

        assert df_check.equals(df)

        return dfs
