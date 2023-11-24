"""
Transformer

Creates cross validation data for models assessment.
It is built on the top of time_series_window transformer.

The usage is documented in the notebook notebooks/documentation/time_series_cv_documentation.py
"""
from datetime import datetime, timedelta
from typing import Any, List
from typing import Union, Tuple

from numpy import array, ndarray, dtype
from numpy import concatenate, float64, float32
from pandas import DataFrame
from pandas import DatetimeIndex
# pylint: disable=no-name-in-module
from pandas._libs.tslibs.timestamps import Timestamp
# pylint: enable=no-name-in-module
from sklearn.utils import shuffle

from src.data.income_weather_data_generator import IncomeWeatherDataGenerator, ATTR_DATE, ATTR_TEMPERATURE, \
    ATTR_RANDOM, ATTR_OUTPUT
from src.exceptions.data_exception import MismatchedDimension, WrongSorting
from src.transformations.base_transformer import BaseTransformer, TransformerDescription
from src.transformations.many_to_datetime_index_transformer import ManyToDatetimeIndexTransformer
from src.transformations.time_series_windows import TimeSeriesWindowsNumpy


def create_weather_data() -> Tuple[DatetimeIndex, List[Union[Timestamp, datetime]], ndarray[Any, dtype[Any]], \
                                   ndarray[Any, dtype[Any]]]:
    """
    Creates data set from income weather data.
    Please see time_series_cv_documentation notebook for more information.
    :return: List[Union[Timestamp, datetime]], ndarray[Any, dtype[Any]]]. (date_time, split_dates, data).
    """
    start_date = "2018-01-01"
    n = 40
    betas = [30, 2, 1, 4, 3, 6, -1, -3, 0, -10, 25, 10]
    sigma = 10

    data_gen = IncomeWeatherDataGenerator()
    df_data, _, _, _ = data_gen.generate(start_date, betas, n, sigma)
    date_time = DatetimeIndex(df_data[ATTR_DATE])
    split_dates = [datetime(2018, 1, 22, 0, 0, 0)]
    X_data = df_data[[ATTR_TEMPERATURE, ATTR_RANDOM]].to_numpy()
    Y_data = df_data[[ATTR_OUTPUT]].to_numpy()

    return date_time, split_dates, X_data, Y_data


def create_sample_data(n: int = 31) -> Tuple[DatetimeIndex, List[Union[Timestamp, datetime]], \
                                             ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], DatetimeIndex]:
    """
    Creates sample data set for tests.
    Please see time_series_cv_documentation notebook for more information.
    :param n: int. Number of observation from 1 to 31 to have meaning. If outside, then put 31 there.
    :return: Tuple[DatetimeIndex, List[Union[Timestamp, datetime]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]],
             DatetimeIndex]. (date_time,
    split_dates, X_data, Y_data, date_time_not_sorted).
    """
    if not 0 < n < 32:
        n = 31
    dates = []
    X_data_list = []
    Y_data_list = []
    for i in range(1, n + 1):
        dates.append(datetime(2000, 1, i, 1, i, i))
        X_data_list.append([i, 10 * i])
        Y_data_list.append([i * 100])
    dates.reverse()

    date_time = DatetimeIndex(dates).sort_values()
    split_dates = [date_time[6], datetime(2000, 1, 15, 0, 0, 0), Timestamp('2000-01-22 0:0:0')]
    X_data: ndarray[Any, dtype[Any]] = array(X_data_list)
    Y_data: ndarray[Any, dtype[Any]] = array(Y_data_list)
    date_time_not_sorted = DatetimeIndex(dates)

    return date_time, split_dates, X_data, Y_data, date_time_not_sorted


# pylint: disable=invalid-name
class TimeSeriesCV(BaseTransformer):  # type:ignore
    """
    NOTE: For better understanding, please see the documentation notebook time_series_cv_documentation.py.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(input_type=[array], input_elements_type=None,
                                                         output_type=[array], output_elements_type=None)
        BaseTransformer.__init__(self, class_name="TimeSeriesCV",
                                 transformer_description=transformer_description)
        self._window_creator = TimeSeriesWindowsNumpy()

        self._X_splits: List[ndarray[Any, dtype[Any]]]
        self._Y_splits: List[ndarray[Any, dtype[Any]]]
        self._cv_splits: List[Tuple[Any, Any]]

    def get_splits(self) -> Tuple[List[ndarray[Any, dtype[Any]]], List[ndarray[Any, dtype[Any]]]]:
        """
        Returns only the splits for time intervals/splits.
        :return: Tuple[List[ndarray[Any, dtype[Any]]], List[ndarray[Any, dtype[Any]]]]. (X_splits, Y_splits)
        """
        return self._X_splits, self._Y_splits

    @staticmethod
    def _test_data_consistency(date_time: DatetimeIndex, X_data: ndarray[Any, dtype[Any]], Y_data: \
            ndarray[Any, dtype[Any]]) -> None:
        """
        Tests basic consistency of the data.
        :param date_time: DatetimeIndex. Timestamp corresponding to the data points in X (and Y if given).
        :param X_data: ndarray[Any, dtype[Any]]. Two-dimensional array. First dimension has to match the length of
                       date_time variable.
        :param Y_data: ndarray[Any, dtype[Any]]. Two-dimensional array. First dimension has to match the length of
                       date_time variable. Y_data can be the same as X_data to get "past", "future" of the dat set.
        """
        pom = date_time.sort_values()
        if sum(pom != date_time):
            raise WrongSorting
        if len(X_data.shape) != 2 or len(Y_data.shape) != 2:
            raise MismatchedDimension("Both X_data and Y_data has to have dimensions equal 2.")
        if X_data.shape[0] != Y_data.shape[0] or X_data.shape[0] != len(date_time):
            raise MismatchedDimension("First dimension of X_data and Y_data has to be the same as len of date_time.")

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def fit(self, date_time: DatetimeIndex, split_dates: List[Union[Timestamp, datetime]], X_data: \
            ndarray[Any, dtype[Any]], Y_data: ndarray[Any, dtype[Any]], Y_same_observation: bool = False, \
            input_window_len: int = 1, output_window_len: int = 1, shift: int = 1, shuffle_data: bool = True, \
            shuffle_seed: int = 8792) -> None:
        """
        Fits the transformation based on the values.
        The date_time has to be SORTED. The data, X and Y, has to be sorted based on date_time as well.
        For better understanding, please see the documentation notebook time_series_cv_documentation.py
        :param date_time: DatetimeIndex. Timestamp corresponding to the data points in X and Y.
        :param split_dates: Union[Timestamp, datetime]. Dates for splits in the date_time. EDGES ARE ADDED
                            AUTOMATICALLY. This is not very good to mix format, but both work. So I keep them
                            both in tests.
        :param X_data: ndarray[Any, dtype[Any]]. Two-dimensional array. First dimension has to match the length of
                       date_time variable.
        :param Y_data: ndarray[Any, dtype[Any]]. Two-dimensional array. First dimension has to match the length of
                       date_time variable. Y_data can be the same as X_data to get "past", "future" of the dat set.
        :param Y_same_observation: bool. If True, that the Y output will start with the same observation (time stamp).
               If False, then it will start with the following observation.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 1 means consecutive data sets are created.
        :param shuffle_data: bool. If to shuffle the data.
        :param shuffle_seed: int. Seed for shuffle.
        """
        # do initial tests
        self._test_data_consistency(date_time, X_data, Y_data)

        # add first and last + one day observation to have full list of dates
        split_dates = [date_time[0]] + split_dates + [date_time[-1] + timedelta(days=1)]

        # create observations pairs for all time intervals (splits).
        self._X_splits = []
        self._Y_splits = []
        for date_from, date_to in zip(split_dates[0:-1], split_dates[1:]):
            take = (date_from <= date_time) & (date_time < date_to)
            X_one_split_data = X_data[take,]
            Y_one_split_data = Y_data[take,]

            if Y_same_observation:
                X_one_split_data = concatenate((X_one_split_data, X_one_split_data[-1:, :]), axis=0)
                Y_one_split_data = concatenate((Y_one_split_data[0:1, :], Y_one_split_data), axis=0)

            X, _ = self._window_creator.fit_predict(X_one_split_data, input_window_len, output_window_len, shift)
            _, Y = self._window_creator.fit_predict(Y_one_split_data, input_window_len, output_window_len, shift)

            # first shuffle for smaller cases
            if shuffle_data:
                X, Y = shuffle(X, Y, random_state=shuffle_seed)

            self._X_splits.append(X)
            self._Y_splits.append(Y)

        # concatenate all except one for k-fold cross validation
        self._cv_splits = []
        if len(split_dates) > 2:
            for i, (X_test, Y_test) in reversed(list(enumerate(zip(self._X_splits, self._Y_splits)))):
                # reverse is there to hold off the last part first
                X_train: ndarray[Any, dtype[Any]] = concatenate([d for j, d in enumerate(self._X_splits) if i != j], \
                                                                axis=0)
                Y_train: ndarray[Any, dtype[Any]] = concatenate([d for j, d in enumerate(self._Y_splits) if i != j], \
                                                                axis=0)

                # another shuffle on the higher level
                if shuffle_data:
                    self._cv_splits.append((
                        shuffle(X_train, Y_train, random_state=shuffle_seed),
                        shuffle(X_test, Y_test, random_state=shuffle_seed)
                    ))
                else:
                    self._cv_splits.append(((X_train, Y_train), (X_test, Y_test)))
        else:
            self._cv_splits.append(((self._X_splits[0], self._Y_splits[0]), (None, None)))

    def fit_predict(self, date_time: DatetimeIndex, split_dates: List[Union[Timestamp, datetime]], X_data: \
            ndarray[Any, dtype[Any]], Y_data: ndarray[Any, dtype[Any]], Y_same_observation: bool = False, \
                    input_window_len: int = 1, output_window_len: int = 1, shift: int = 1, shuffle_data: bool = True, \
                    shuffle_seed: int = 8792) -> List[Tuple[Any, Any]]:
        """
        Fits and predicts.
        The date_time has to be SORTED. The data, X and Y, has to be sorted based on date_time as well.
        For better understanding, please see the documentation notebook time_series_cv_documentation.py
        :param date_time: DatetimeIndex. Timestamp corresponding to the data points in X and Y.
        :param split_dates: Union[Timestamp, datetime]. Dates for splits in the date_time. EDGES ARE ADDED
                            AUTOMATICALLY. This is not very good to mix format, but both work. So I keep them
                            both in tests.
        :param X_data: ndarray[Any, dtype[Any]]. Two-dimensional array. First dimension has to match the length of
                       date_time variable.
        :param Y_data: ndarray[Any, dtype[Any]]. Two-dimensional array. First dimension has to match the length of
                       date_time variable. Y_data can be the same as X_data to get "past", "future" of the dat set.
        :param Y_same_observation: bool. If True, that the Y output will start with the same observation (time stamp).
               If False, then it will start with the following observation.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :param shuffle_data: bool. If to shuffle the data.
        :param shuffle_seed: int. Seed for shuffle.
        :return: List[Tuple[Any, Any]]. List of tuples of tuples [((X_train, Y_train), (X_test, Y_test))].
        """
        self.fit(date_time, split_dates, X_data, Y_data, Y_same_observation, input_window_len, output_window_len, shift,
                 shuffle_data, shuffle_seed)
        return self.predict()

    def predict(self) -> List[Tuple[Any, Any]]:
        """
        Predicts the output.
        :return: List[Tuple[Any, Any]]. List of tuples of tuples [((X_train, Y_train), (X_test, Y_test))].
        """
        return self._cv_splits

    def inverse(self) -> None:
        """
        Does the inverse transformation.
        """
        return None
    # pylint: enable=arguments-differ
    # pylint: enable=too-many-arguments
    # pylint: enable=too-many-locals
    # pylint: enable=invalid-name


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def create_windows_from_list_of_data_frames(dfs: List[DataFrame], attr_date_time: str, attrs_X: List[str], \
                                            attrs_Y: List[str], y_same_observation: bool, input_window_len: int,
                                            output_window_len: int, shift: int, \
                                            shuffle_data: bool, shuffle_seed: int) -> Tuple[
    ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    """
    Create windows for X and Y data from each data frame and then concatenate them into one array.
    :param dfs: List[DataFrame].
    :param attr_date_time: str.
    :param attrs_X: List[str].
    :param attrs_Y: List[str].
    :param Y_same_observation: bool. If True, that the Y output will start with the same observation (time stamp).
           If False, then it will start with the following observation.
    :param input_window_len: int. Length of the new inputs window.
    :param output_window_len: int. Length of the new outputs window.
    :param shift: int. Shift between the data sets. 1 means consecutive data sets are created.
    :param shuffle_data: bool. If to shuffle the data.
    :param shuffle_seed: int. Seed for shuffle.
    :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. <X, Y>.
    """
    ts_cv = TimeSeriesCV()
    to_date_time_tr = ManyToDatetimeIndexTransformer()
    X_arrays = []
    Y_arrays = []
    for df in dfs:
        cross_validation_results = ts_cv.fit_predict(
            date_time=to_date_time_tr.fit_predict(list(df[attr_date_time])),
            split_dates=[],
            X_data=df[attrs_X].to_numpy(dtype=float32),
            Y_data=df[attrs_Y].to_numpy(dtype=float64),
            Y_same_observation=y_same_observation,
            input_window_len=input_window_len,
            output_window_len=output_window_len,
            shift=shift,
            shuffle_data=shuffle_data,
            shuffle_seed=shuffle_seed
        )
        X_arrays.append(cross_validation_results[0][0][0])
        Y_arrays.append(cross_validation_results[0][0][1])

    return concatenate(X_arrays), concatenate(Y_arrays)


# pylint: enable=too-many-arguments
# pylint: enable=too-many-locals

def split_list_of_data_frames(dfs: List[DataFrame], attr_date_time: str, train_end: datetime, dev_end: datetime,
                              attrs: List[str], n_min: int) -> Tuple[List[DataFrame], List[DataFrame], List[DataFrame]]:
    """
    Takes a list of data frames and splits it into three lists of data frames taking observations from those data frames
    under the train_end (train data), then between train_end and val_end (validation data), and bigger than val_end
    (train data). Takes only data frames with more than or equal n_min observations.
    :param dfs: List[DataFrame]. List of data frames to be split.
    :param attr_date_time: str. Attribute with date time index.
    :param train_end: datetime. End of train period.
    :param dev_end: datetime. End of validation period.
    :param attrs: List[str]. List of attributes to be taken.
    :param n_min:
    :return: Tuple[List[DataFrame], List[DataFrame], List[DataFrame]]. dfs_train, dfs_val, dfs_test.
    """
    dfs_train = []
    dfs_val = []
    dfs_test = []

    for df in dfs:
        date_time = df[attr_date_time]

        # train --------------------------------------------------------------------------------------------------------
        take = date_time < train_end
        df_pom = df.loc[take, attrs]
        if df_pom.shape[0] >= n_min:
            dfs_train.append(df_pom)

        # val ----------------------------------------------------------------------------------------------------------
        take = (train_end <= date_time) & (date_time < dev_end)
        df_pom = df.loc[take, attrs]
        if df_pom.shape[0] >= n_min:
            dfs_val.append(df_pom)

        # test ---------------------------------------------------------------------------------------------------------
        take = dev_end <= date_time
        df_pom = df.loc[take, attrs]
        if df_pom.shape[0] >= n_min:
            dfs_test.append(df_pom)

    return dfs_train, dfs_val, dfs_test


def get_n_of_observations(dfs_list: List[DataFrame]) -> int:
    """
    Gets the sum of rows of all data frames.
    :param dfs_list: List[DataFrame]. List of data frames.
    :return: int. N of observations.
    """
    n = 0
    for df in dfs_list:
        n += df.shape[0]
    return n
