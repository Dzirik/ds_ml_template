"""
Saver and loader

Class for simplification of saving and loading.
"""
import os
import pickle
from typing import Any

from pandas import DataFrame, read_pickle, read_csv

from src.utils.config import Config


def get_path(file_name: str, where: str = "raw_data", extension: str = ".pkl") -> str:
    """
    Returns a path of a file from the config file based of where.
    :param file_name: str. File name without .pkl.
    :param where: str. Name of the path from the config file.
    :param extension: str. File extension. Default is .pkl.
    :return: str. Path as string.
    """
    dir_path = getattr(Config().get().path, where)

    return os.path.abspath(os.path.join(dir_path, file_name + extension))


class SaverAndLoader():
    """
    Class for simplification of saving and loading.
    """

    def __init__(self) -> None:
        self._decimal = ","
        self._sep = ";"

    def set_csv_params(self, decimal: str, sep: str) -> None:
        """
        Sets the parameters for read_csv function.
        :param decimal: str.
        :param sep: str.
        """
        self._decimal = decimal
        self._sep = sep

    def save_dataframe_to_csv(self, df: DataFrame, file_name: str, where: str = "raw_data") -> None:
        """
        Saves a DataFrame to a csv file.
        :param df: DataFrame to be saved.
        :param file_name: str. File name without .csv
        :param where: str. Name of the path from the config file.
        """
        path = get_path(file_name, where, ".csv")
        df.to_csv(path, index=False, decimal=self._decimal, sep=self._sep)

    def load_dataframe_from_csv(self, file_name: str, where: str = "raw_data", min_size: int = 2) -> DataFrame:
        """
        Saves a DataFrame to a csv file.
        :param file_name: str. File name without .csv
        :param where: str. Name of the path from the config file.
        :param min_size: int. Minimum size of the file.
        :return: DataFrame
        """
        path = get_path(file_name, where, ".csv")
        if os.path.exists(path):
            # zero len doesnt work with csv files in cry project, it had length 2.
            if os.path.getsize(path) > min_size:
                return read_csv(path, decimal=self._decimal, sep=self._sep)
        return DataFrame()

    @staticmethod
    def save_dataframe_to_pickle(df: DataFrame, file_name: str, where: str = "raw_data") -> None:
        """
        Saves a DataFrame to a pickle file.
        :param df: DataFrame to be saved.
        :param file_name: str. File name without .pkl.
        :param where: str. Name of the path from the config file.
        """
        path = get_path(file_name, where)
        df.to_pickle(path, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_dataframe_from_pickle(file_name: str, where: str = "raw_data", min_size: int = 0) -> DataFrame:
        """
        Loads a DataFrame from a pickle file.
        :param file_name: str. File name without .pkl.
        :param where: str. Name of the path from the config file.
        :param min_size: int. Minimum size of the file.
        :return: DataFrame. Loaded data frame
        """
        path = get_path(file_name, where)
        if os.path.exists(path):
            if os.path.getsize(path) > min_size:
                df = read_pickle(path)
                return df
        return DataFrame()

    @staticmethod
    def save_to_pickle(data: Any, file_name: str, where: str = "raw_data") -> None:
        """
        Saves data to pickle file.
        :param data: Any. Python data to be saved.
        :param file_name: str. File name without .pkl.
        :param where: str. Name of the path from the config file.
        """
        path = get_path(file_name, where)

        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_pickle(file_name: str, where: str = "raw_data", min_size: int = 0) -> Any:
        """
        Loads a pickle file.
        :param file_name: str. File name without .pkl.
        :param where: str. Name of the path from the config file.
        :param min_size: int. Minimum size of the file.
        :return: Any. Loaded data from the pickle file.
        """
        path = get_path(file_name, where)
        if os.path.exists(path):
            if os.path.getsize(path) > min_size:
                with open(path, 'rb') as handle:
                    loaded = pickle.load(handle)
                return loaded
        return None

    @staticmethod
    def delete_pickle(file_name: str, where: str = "raw_data") -> None:
        """
        Deletes a pickle file.
        :param file_name: str. File name without .pkl.
        :param where: str. Name of the path from the config file.
        """
        path = get_path(file_name, where)
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    SAVER_AND_LOADER: SaverAndLoader
    SAVER_AND_LOADER = SaverAndLoader()

    DATA_FRAME = DataFrame([[1, 1], [2, 2]])
    DATA_FRAME_FILE_NAME = "test_data_frame"
    SAVER_AND_LOADER.save_dataframe_to_pickle(DATA_FRAME, DATA_FRAME_FILE_NAME)
    print(f"Loaded: {SAVER_AND_LOADER.load_dataframe_from_pickle(DATA_FRAME_FILE_NAME)}")

    MY_LIST = [1, 2, 3]
    MY_LIST_NAME = "test_list"
    SAVER_AND_LOADER.save_to_pickle(MY_LIST, MY_LIST_NAME)
    print(f"Loaded: {SAVER_AND_LOADER.load_from_pickle(MY_LIST_NAME)}")

    print(f"Loaded Wrong Named DF: {SAVER_AND_LOADER.load_dataframe_from_pickle('wrong')}")

    SAVER_AND_LOADER.delete_pickle(DATA_FRAME_FILE_NAME)
    SAVER_AND_LOADER.delete_pickle(MY_LIST_NAME)
