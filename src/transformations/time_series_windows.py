"""
Transformer

Creates training/testing windows of selected length for time series as 2D object (time x data dim). There are three
variants:
- Dummy - just for testing.
- Numpy. - the fastest.
- Pandas.

Testing and comparison can be found inf notebooks/documentation/time_series_windows_documentation.py
"""

from typing import Tuple, Any, List

from numpy import array, zeros, hsplit, lib, concatenate, ndarray, dtype
from pandas import DataFrame

from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class TimeSeriesWindowsDummy(BaseTransformer):  # type:ignore
    """
    Creates training/testing windows of selected length for time series as 2D array (time x data dim)
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(input_type=[array], input_elements_type=None,
                                                         output_type=[array], output_elements_type=None)
        BaseTransformer.__init__(self, class_name="TimeSeriesWindowsDummy",
                                 transformer_description=transformer_description)

    @staticmethod
    def _transform(data: ndarray[Any, dtype[Any]], input_window_len: int, output_window_len: int, shift: int) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Creates two arrays with windows.
        :param data: ndarray[Any, dtype[Any]]. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. Tuple of input data list and output data
                 list.
        """
        n = data.shape[0]
        p = data.shape[1]

        window_width = input_window_len + output_window_len

        n_out = len(list(range(0, n - window_width + 1, shift)))
        X = zeros((n_out, input_window_len, p))
        Y = zeros((n_out, output_window_len, p))

        for i_out, i_in in enumerate(list(range(0, n - window_width + 1, shift))):
            for j in range(input_window_len):
                X[i_out, j,] = data[i_in + j,]

            for j in range(output_window_len):
                Y[i_out, j,] = data[i_in + input_window_len + j,]

        return X, Y

    # pylint: disable=arguments-differ
    def fit(self, data: ndarray[Any, dtype[Any]], input_window_len: int, output_window_len: int, shift: int) -> None:
        """
        Fits the transformation based on the values.
        :param data: ndarray[Any, dtype[Any]]. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[array, array]. Tuple of input data list and output data list.
        """

    def fit_predict(self, data: ndarray[Any, dtype[Any]], input_window_len: int, output_window_len: int, shift: int) \
            -> Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Fits and predicts.
        :param data: ndarray[Any, dtype[Any]]. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. Tuple of input data list and output data
                 list.
        """
        return self._transform(data, input_window_len, output_window_len, shift)

    def predict(self, data: ndarray[Any, dtype[Any]], input_window_len: int, output_window_len: int, shift: int) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Predicts the output.
        :param data: ndarray[Any, dtype[Any]]. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. Tuple of input data list and output data
                 list.
        """
        return self._transform(data, input_window_len, output_window_len, shift)
    # pylint: enable=arguments-differ


class TimeSeriesWindowsNumpy(BaseTransformer):  # type:ignore
    """
    Creates training/testing windows of selected length for time series as 2D array (time x data dim).

    This numpy variant can't do only 1D array. For that one, the TimeSeriesWindowsDummy is used.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(input_type=[array], input_elements_type=None,
                                                         output_type=[array], output_elements_type=None)
        BaseTransformer.__init__(self, class_name="TimeSeriesWindowsNumpy",
                                 transformer_description=transformer_description)

    @staticmethod
    def _transform(data: ndarray[Any, dtype[Any]], input_window_len: int, output_window_len: int, shift: int) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Creates two arrays with windows.
        :param data: ndarray[Any, dtype[Any]]. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. Tuple of input data list and output data
                 list.
        """
        # the approach can't work with one dime array, for that, dummy is used.
        if data.shape[1] == 1:
            transformer = TimeSeriesWindowsDummy()
            return transformer.fit_predict(data, input_window_len, output_window_len, shift)

        # numpy variant for p > 1
        window_width = input_window_len + output_window_len
        n = data.shape[0]

        array_for_indices: ndarray[Any, dtype[Any]]
        if data.shape[1] > 1:
            array_for_indices = array(list(range(n))).reshape(-1, 1)
        else:
            array_for_indices = data
            data = array([])

        shape = (array_for_indices.size - window_width + 1, window_width)
        strides = (array_for_indices.itemsize, array_for_indices.itemsize)
        indices: ndarray[Any, dtype[Any]] = lib.stride_tricks.as_strided(array_for_indices, shape=shape, \
                                                                         strides=strides)

        output: List[ndarray[Any, dtype[Any]]]
        if data is None:
            output = hsplit(indices, (input_window_len, window_width))
        else:
            output = hsplit(data[indices,], (input_window_len, window_width))

        rows_to_take = list(range(0, n - window_width + 1, shift))
        return output[0][rows_to_take], output[1][rows_to_take]

    # pylint: disable=arguments-differ
    def fit(self, data: ndarray[Any, dtype[Any]], input_window_len: int, output_window_len: int, shift: int) -> None:
        """
        Fits the transformation based on the values.
        :param data: ndarray[Any, dtype[Any]]. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[array, array]. Tuple of input data list and output data list.
        """

    def fit_predict(self, data: ndarray[Any, dtype[Any]], input_window_len: int, output_window_len: int, shift: int) \
            -> Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Fits and predicts.
        :param data: ndarray[Any, dtype[Any]]. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. Tuple of input data list and output data
                 list.
        """
        return self._transform(data, input_window_len, output_window_len, shift)

    def predict(self, data: ndarray[Any, dtype[Any]], input_window_len: int, output_window_len: int, shift: int) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Predicts the output.
        :param data: ndarray[Any, dtype[Any]]. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. Tuple of input data list and output data
                 list.
        """
        return self._transform(data, input_window_len, output_window_len, shift)
    # pylint: enable=arguments-differ


class TimeSeriesWindowsPandas(BaseTransformer):  # type:ignore
    """
    Creates training/testing windows of selected length for time series as DataFrame (time x data dim).
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(input_type=[DataFrame], input_elements_type=None,
                                                         output_type=[array], output_elements_type=None)
        BaseTransformer.__init__(self, class_name="TimeSeriesWindowsPandas",
                                 transformer_description=transformer_description)

    # pylint: disable=invalid-name
    @staticmethod
    def _transform(df: DataFrame, input_window_len: int, output_window_len: int, shift: int) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Creates two arrays with windows.
        :param data: DataFrame. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. Tuple of input data list and output data
                 list.
        """
        n = df.shape[0]
        p = df.shape[1]
        window_width = input_window_len + output_window_len
        X_df = None
        Y_df = None
        for window in df.rolling(window=window_width):
            if window.shape[0] >= window_width:
                if X_df is None:
                    X_df = window.to_numpy()[0:input_window_len, ].reshape((1, input_window_len, p))
                    Y_df = window.to_numpy()[input_window_len:window_width, ].reshape((1, output_window_len, p))
                else:
                    X_df = concatenate(
                        (X_df, window.to_numpy()[0:input_window_len, ].reshape((1, input_window_len, p))))
                    Y_df = concatenate(
                        (Y_df, window.to_numpy()[input_window_len:window_width, ].reshape((1, output_window_len, p))))

        rows_to_take = list(range(0, n - window_width + 1, shift))

        return X_df[rows_to_take,], Y_df[rows_to_take,]  # type:ignore

    # pylint: enable=invalid-name

    # pylint: disable=arguments-differ
    def fit(self, df: DataFrame, input_window_len: int, output_window_len: int, shift: int) -> None:
        """
        Fits the transformation based on the values.
        :param data: DataFrame. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[array, array]. Tuple of input data list and output data list.
        """

    def fit_predict(self, df: DataFrame, input_window_len: int, output_window_len: int, shift: int) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Fits and predicts.
        :param data: DataFrame. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. Tuple of input data list and output data
                 list.
        """
        return self._transform(df, input_window_len, output_window_len, shift)

    def predict(self, df: DataFrame, input_window_len: int, output_window_len: int, shift: int) -> \
            Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Predicts the output.
        :param data: DataFrame. Two dimensional array of data.
        :param input_window_len: int. Length of the new inputs window.
        :param output_window_len: int. Length of the new outputs window.
        :param shift: int. Shift between the data sets. 0 means consecutive data sets are created.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]. Tuple of input data list and output data
                 list.
        """
        return self._transform(df, input_window_len, output_window_len, shift)
    # pylint: enable=arguments-differ
