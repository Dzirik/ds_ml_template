"""
Pipeline.
"""
from builtins import str
from typing import Optional

from pandas import DataFrame

from src.data.attributes import PREPOSITION_DIFF_PERC, PREPOSITION_DIFF
from src.data.saver_and_loader import SaverAndLoader
from src.exceptions.development_exception import NoProperOptionInIf
from src.pipelines.base_pipeline import BasePipeline
from src.pipelines.new_attributes_pipeline_config import NewAttributesPipelineConfig
from src.transformations.datetime_one_hot_transformer import DatetimeOneHotEncoderTransformer
from src.transformations.difference_percentage_change_transformer import DifferencePercentageChangeTransformer
from src.transformations.difference_transformer import DifferenceTransformer
from src.utils.date_time_functions import convert_datetime_to_string_date


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-branches
class NewAttributesPipeline(BasePipeline):  # type: ignore
    """
    Creates new attributes.
    """

    def __init__(self, config_file_name: Optional[str] = None) -> None:
        BasePipeline.__init__(self, class_name="NewAttributesPipeline",
                              config_file_name=config_file_name,
                              config_class=NewAttributesPipelineConfig)

        self._saver_and_loader = SaverAndLoader()

        self._diff_tr = DifferenceTransformer()
        self._diff_perc_tr = DifferencePercentageChangeTransformer()
        self._oh_tr = DatetimeOneHotEncoderTransformer()

        # for testing purpose
        self._oh_t_file_name: str = ""

    def _execute_one_data_frame(self, df: DataFrame) -> DataFrame:
        """
        Creates new attributes for one data frame..
        :param df: DataFrame. Data frame to be transformed.
        :return: DataFrame. Original data frame with new attributes from configuration.
        """
        for conf in self._config_data.new_attributes:
            if conf.create:
                if conf.name_of_transformation == "diff":
                    for attr_name in conf.for_attrs:
                        data = df[attr_name].to_numpy().reshape((df.shape[0], 1))
                        df[f"{attr_name}_{PREPOSITION_DIFF}"] = self._diff_tr.fit_predict(
                            data, **conf.parameters
                        )
                elif conf.name_of_transformation == "diff_perc":
                    for attr_name in conf.for_attrs:
                        data = df[attr_name].to_numpy().reshape((df.shape[0], 1))
                        df[f"{attr_name}_{PREPOSITION_DIFF_PERC}"] = self._diff_perc_tr.fit_predict(
                            data, **conf.parameters
                        )
                elif conf.name_of_transformation == "datetime_family":
                    if conf.use_saved:
                        oh_tr = self._saver_and_loader.load_from_pickle(file_name=conf.name_of_file, where="models")

                        oh_data = oh_tr.predict(df.index)
                        oh_attr_names = oh_tr.get_encoded_attribute_names()
                    else:
                        oh_data = self._oh_tr.fit_predict(df.index, **conf.parameters)
                        oh_attr_names = self._oh_tr.get_encoded_attribute_names()

                        self._oh_t_file_name = f"{convert_datetime_to_string_date()}_oh_t_datetime"
                        self._saver_and_loader.save_to_pickle(data=self._oh_tr, file_name=self._oh_t_file_name,
                                                              where="models")
                    for i, attr in enumerate(oh_attr_names):
                        df[attr] = oh_data[:, i]
                else:
                    print(conf.name_of_transformation)
                    raise NoProperOptionInIf
        return df

    def get_oh_t_file_name(self) -> str:
        """
        Gets the oh transformer saved file name.
        :return: str.
        """
        return self._oh_t_file_name
# pylint: enable=too-many-instance-attributes
# pylint: enable=too-many-branches
