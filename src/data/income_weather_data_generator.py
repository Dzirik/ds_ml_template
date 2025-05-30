"""
Data Generator

File for generating regression _data simulating income based on the date and weather.
The documentation can be found in notebook/documentation/income_weather_data_generator_documentation.py.
"""
from datetime import datetime, timedelta
from typing import Tuple, List
from typing import Union, Any

from numpy import ndarray, dtype, array, zeros, arange
from numpy import random
from numpy import sum as summation
from pandas import DataFrame, get_dummies
from pandas import Series

from src.transformations.time_series_windows import TimeSeriesWindowsNumpy

ATTR_DATE = "DATE"
ATTR_DAY_OF_WEEK_NUM = "DAY_OF_WEEK_NUM"
ATTR_WEEK_NUM = "WEEK_NUM"
ATTR_WEATHER = "WEATHER"
ATTR_TEMPERATURE = "TEMPERATURE"
ATTR_RANDOM = "RANDOM"
ATTR_OUTPUT = "OUTPUT"
WEATHER_TYPES = ["sun", "rain", "wind", "cloud"]


# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
class IncomeWeatherDataGenerator:
    """
    Class for generating the _data.
    """

    def __init__(self) -> None:
        self._time_series_window_transformer = TimeSeriesWindowsNumpy()
        self._n: int
        self._data = DataFrame()
        self._data_transformed = DataFrame()
        self._X_multi: ndarray[Any, dtype[Any]]
        self._Y_multi: ndarray[Any, dtype[Any]]
        self._seed = 39206
        self._weights_multi: ndarray[Any, dtype[Any]]

    def set_seed(self, seed_number: int) -> None:
        """
        Sets _seed number.
        :param seed_number: int. New _seed
        """
        self._seed = seed_number

    def _create_dates(self, start_date: str) -> None:
        """
        Creates three columns with the date, day of the week number and week number of each observations.
        :param start_date: str. Starting date of the observations.
        """
        start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_datetime = start_date_datetime + timedelta(days=self._n - 1)
        delta = timedelta(days=1)
        d = start_date_datetime
        dates = []
        while d <= end_date_datetime:
            dates.append(d)
            d += delta
        self._data[ATTR_DATE] = dates
        self._data[ATTR_DAY_OF_WEEK_NUM] = [int(d.strftime("%w")) for d in dates]
        self._data[ATTR_WEEK_NUM] = [int(d.strftime("%V")) for d in dates]

    def _create_categorical(self) -> None:
        """
        Creates a column with a random values chosen from the list of strings WEATHER_TYPES.
        """
        self._data[ATTR_WEATHER] = random.choice(WEATHER_TYPES, self._n, replace=True)

    def _create_temperature(self) -> None:
        """
        Creates a column with a random value.
        """
        self._data[ATTR_TEMPERATURE] = random.uniform(10, 35, self._n)

    def _create_random(self) -> None:
        """
        Creates a column with a random value.
        """
        self._data[ATTR_RANDOM] = random.randn(self._n) * 10

    def save_as_csv(self) -> None:
        """
        Saves the _data as a CSV file.
        """
        self._data.to_csv("_data.csv", index=False)

    def get_weights_multi(self) -> ndarray[Any, dtype[Any]]:
        """
        Returns the weights used in multi dimensional example.
        :return: ndarray[Any, dtype[Any]].
        """
        return self._weights_multi

    def get_attributes_names_multi(self) -> List[str]:
        """
        Returns the names of attributes in multi dimensional example.
        :return: List[str]. List of attribute name.
        """
        return list(self._data_transformed.columns)[0:13]

    def _create_multi_dim(self, input_window_len: int = 20) -> None:
        """
        Creates multidimensional regression.
        :param input_window_length: int.
        """
        df = self._data_transformed.copy()
        df.drop(columns=["OUTPUT"], inplace=True)
        # I need only X, Y is calculated as another list.
        self._X_multi, _ = self._time_series_window_transformer.fit_predict(
            data=df.to_numpy(), input_window_len=input_window_len, output_window_len=0, shift=1
        )
        random.seed(self._seed)
        self._weights_multi = random.randint(low=-20, high=100, size=self._X_multi[0, :, :].shape)
        # self._Y_multi = [nan] * (self._X_multi[0].shape[0] - 1)
        list_pom = []
        for i in range(self._X_multi.shape[0]):
            Z = self._X_multi[i, :, :] * self._weights_multi
            list_pom.append(Z.sum())
        n = len(list_pom)
        self._Y_multi = array(list_pom)
        self._Y_multi = self._Y_multi.reshape((n, 1))

    @staticmethod
    def _create_classes_for_multidim_ml_data(Y: Series, bin_mean: float, ter_25: float, ter_75: float) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], \
                  ndarray[Any, dtype[Any]]]:
        """
        Creates the binary output attribute above threshold Y_bin, dense tertiary attribute Y_ter and sparse
        Y_ter_oh for the output variable.
        :param Y: Series. Series of the output.
        :param bin_mean: float. Mean for binary.
        :param ter_25: float. Lower bound for tertiary.
        :param ter_75: float. Upper bound for tertiary.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], \
                    ndarray[Any, dtype[Any]]]. (Y_bin, Y_ter, Y_ter_oh).
        """
        # regression
        Y_reg: ndarray[Any, dtype[Any]] = Y

        # binary
        Y_bin: ndarray[Any, dtype[Any]] = array(Y_reg > bin_mean).astype(int)

        # terciary
        Y_ter: ndarray[Any, dtype[Any]] = array(Y_reg > ter_25).astype(int) + array(Y_reg > ter_75).astype(int)

        p = 3
        Y_ter_oh: ndarray[Any, dtype[Any]] = zeros((len(Y_bin), p))
        Y_ter_oh[arange(Y_ter.shape[0]), Y_ter.reshape((Y_ter.shape[0],))] = 1

        return Y_reg, Y_bin, Y_ter, Y_ter_oh

    # pylint: disable=too-many-locals
    def generate_multidim_ml_data(self) -> Tuple[Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]],
                                                 Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]],
                                                 Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]], \
                                                 Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]],
                                                 Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]]:
        """
        :return: (train_X, test_X), (train_Y_reg, test_Y_reg), (train_Y_bin, test_Y_bin), (train_Y_ter, test_Y_ter), \
               (train_Y_ter_oh, test_Y_ter_oh)
        """
        start_date = "2000-01-01"
        n = 365 * 15 + 4
        betas = [30., 2, 1, 4, 3, 6, -1, -3, 0, -10, 25, 10]
        sigma = 10
        _, _, train_X_multi, train_Y_multi = self.generate(start_date, betas, n, sigma)

        start_date = "2016-01-01"
        n = 365 * 5 + 2
        betas = [30., 2, 1, 4, 3, 6, -1, -3, 0, -10, 25, 10]
        sigma = 10
        _, _, test_X_multi, test_Y_multi = self.generate(start_date, betas, n, sigma)

        train_Y_reg, train_Y_bin, train_Y_ter, train_Y_ter_oh = self._create_classes_for_multidim_ml_data(
            train_Y_multi, 19000., 17000., 21000.
        )
        test_Y_reg, test_Y_bin, test_Y_ter, test_Y_ter_oh = self._create_classes_for_multidim_ml_data(
            test_Y_multi, 19000., 17000., 21000.
        )

        return (train_X_multi, test_X_multi), (train_Y_reg, test_Y_reg), (train_Y_bin, test_Y_bin), (
            train_Y_ter, test_Y_ter), (train_Y_ter_oh, test_Y_ter_oh)

    # pylint: enable=too-many-locals

    @staticmethod
    def _create_classes_for_basic_ml_data(Y: Series, bin_mean: float, ter_25: float, ter_75: float) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], \
                  ndarray[Any, dtype[Any]]]:
        """
        Creates the binary output attribute above threshold Y_bin, dense tertiary attribute Y_ter and sparse
        Y_ter_oh for the output variable.
        :param Y: Series. Series of the output.
        :param bin_mean: float. Mean for binary.
        :param ter_25: float. Lower bound for tertiary.
        :param ter_75: float. Upper bound for tertiary.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], \
                    ndarray[Any, dtype[Any]]]. (Y_reg, Y_bin, Y_ter, Y_ter_oh).
        """
        # regression
        Y_reg: ndarray[Any, dtype[Any]] = array(Y).reshape((len(Y), 1))

        # binary
        Y_bin: ndarray[Any, dtype[Any]] = array([[1] if x else [0] for x in list(Y > bin_mean)])

        # terciary
        pom_1 = [1 if x else 0 for x in list(Y > ter_25)]
        pom_2 = [1 if x else 0 for x in list(Y > ter_75)]
        Y_ter: ndarray[Any, dtype[Any]] = array([x + y for x, y in zip(pom_1, pom_2)])

        p = 3
        Y_ter_oh: ndarray[Any, dtype[Any]] = zeros((len(Y_bin), p))
        Y_ter_oh[arange(Y_ter.size), Y_ter] = 1

        Y_ter = Y_ter.reshape((len(Y_ter), 1))

        return Y_reg, Y_bin, Y_ter, Y_ter_oh

    # pylint: disable=too-many-locals
    def generate_basic_ml_data(self) -> Tuple[Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]],
                                              Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]],
                                              Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]], \
                                              Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]],
                                              Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]]:
        """
        :return: (train_X, test_X), (train_Y_reg, test_Y_reg), (train_Y_bin, test_Y_bin), (train_Y_ter, test_Y_ter), \
               (train_Y_ter_oh, test_Y_ter_oh)
        """
        start_date = "2000-01-01"
        n = 365 * 15 + 4
        betas = [30., 2, 1, 4, 3, 6, -1, -3, 0, -10, 25, 10]
        sigma = 10
        train_data, train_data_transformed, _, _ = self.generate(start_date, betas, n, sigma)

        start_date = "2016-01-01"
        n = 365 * 5 + 2
        betas = [30., 2, 1, 4, 3, 6, -1, -3, 0, -10, 25, 10]
        sigma = 10
        test_data, test_data_transformed, _, _ = self.generate(start_date, betas, n, sigma)

        train_Y_reg, train_Y_bin, train_Y_ter, train_Y_ter_oh = self._create_classes_for_basic_ml_data(
            train_data[ATTR_OUTPUT], 700., 500., 900.
        )
        test_Y_reg, test_Y_bin, test_Y_ter, test_Y_ter_oh = self._create_classes_for_basic_ml_data(
            test_data[ATTR_OUTPUT], 700., 500., 900.
        )

        train_data_transformed.drop(ATTR_RANDOM, axis=1, inplace=True)
        train_data_transformed.drop(ATTR_OUTPUT, axis=1, inplace=True)
        train_X: ndarray[Any, dtype[Any]] = train_data_transformed.to_numpy()

        test_data_transformed.drop(ATTR_RANDOM, axis=1, inplace=True)
        test_data_transformed.drop(ATTR_OUTPUT, axis=1, inplace=True)
        test_X: ndarray[Any, dtype[Any]] = test_data_transformed.to_numpy()

        return (train_X, test_X), (train_Y_reg, test_Y_reg), (train_Y_bin, test_Y_bin), (train_Y_ter, test_Y_ter), \
               (train_Y_ter_oh, test_Y_ter_oh)

    # pylint: enable=too-many-locals

    def generate(self, start_date: str, betas: List[Union[int, float]], n: int = 100, sigma: int = 10,
                 input_window_len: int = 20) -> Tuple[DataFrame, DataFrame, ndarray[Any, dtype[Any]], \
                                                      ndarray[Any, dtype[Any]]]:
        """
        Creates the _data set which includes an output column comprised by the output of a linear regression.
        :param start_date: str. Starting date of the observations.
        :param betas: List[Union[int, float]]. List of coefficients for attributes for transformed _data frame of length
                12.
                ['TEMPERATURE', 'DAY_OF_WEEK_NUM_0', 'DAY_OF_WEEK_NUM_1',
                'DAY_OF_WEEK_NUM_2', 'DAY_OF_WEEK_NUM_3', 'DAY_OF_WEEK_NUM_4',
                'DAY_OF_WEEK_NUM_5', 'DAY_OF_WEEK_NUM_6', 'WEATHER_cloud',
                'WEATHER_rain', 'WEATHER_sun', 'WEATHER_wind']
                Example: [30.1, 2, 1, 4, 3, 6, -1, -3, 0, -10, 25, 10]
        :param n: int. Number of days to be generated.
        :param sigma: Union[int, float]. Intercept coefficient.
        :param input_window_len: int. Length of the historical data backwards for creating multidimensional
            regression.
        :return: Tuple[DataFrame, DataFrame, List[ndarray[Any, dtype[Any]]], List[ndarray[Any, dtype[Any]]]].
                 Non-tranformed _data frame, transformed _data frame, array of multi dimensional regression X and Y.
        """
        self._data = DataFrame()

        self._n = n

        # generate columns
        random.seed(self._seed)
        self._create_dates(start_date)
        self._create_categorical()
        self._create_temperature()
        self._create_random()

        # transform and compute regression output
        self._data_transformed = get_dummies(data=self._data[["TEMPERATURE", "DAY_OF_WEEK_NUM", "WEATHER"]],  # _data
                                             columns=["DAY_OF_WEEK_NUM", "WEATHER"])  # to be encoded/transformed

        self._data[ATTR_OUTPUT] = summation(self._data_transformed.values * betas, axis=1) + random.randn(n) * sigma

        # delete unused columns in non-transformed _data frame
        del self._data[ATTR_DAY_OF_WEEK_NUM]
        del self._data[ATTR_WEEK_NUM]

        # add missing columns to transformed
        self._data_transformed[ATTR_RANDOM] = self._data[ATTR_RANDOM]
        self._data_transformed[ATTR_OUTPUT] = self._data[ATTR_OUTPUT]

        # create multi data
        self._create_multi_dim(input_window_len)

        return self._data, self._data_transformed, self._X_multi, self._Y_multi
# pylint: enable=invalid-name
# pylint: enable=too-many-instance-attributes
