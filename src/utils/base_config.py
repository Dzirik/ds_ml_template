"""
Class for general interface for configs.
"""
from os.path import join
from typing import Union, Dict, Any

import typedload
from pyhocon import ConfigFactory

from src.constants.global_constants import FOLDER_CONFIGURATIONS
from src.pipelines.columns_grouping_pipeline_config_data import ColumnsGroupingPipelineConfigData
from src.pipelines.row_blocks_grouping_pipeline_config_data import RowBlocksGroupingPipelineConfigData
from src.pipelines.transfomations_executioner_pipeline_config_data import TransformationsExecutionerPipelineConfigData
from src.utils.logger import Logger
from src.utils.meta_class import MetaClass, CONFIG_TYPE_NAME


class BaseConfig(MetaClass):  # type:ignore
    """
    Parent class for general interface for configs.

    Only src/utils/config is different, but interface is the same.
    """

    def __init__(self, class_name: str, config_file_name: str, \
                 data_structure: Union[RowBlocksGroupingPipelineConfigData, ColumnsGroupingPipelineConfigData,
                                       TransformationsExecutionerPipelineConfigData]) -> None:
        MetaClass.__init__(self, class_type=CONFIG_TYPE_NAME, class_name=class_name)

        self._config_file_name = config_file_name
        self._data_structure = data_structure
        self._data: Union[RowBlocksGroupingPipelineConfigData, ColumnsGroupingPipelineConfigData,
                          TransformationsExecutionerPipelineConfigData]

        self.parse_config()

    def parse_config(self) -> None:
        """
        Parses the config.
        """
        config_file_path = join("../../", FOLDER_CONFIGURATIONS, f"{self._config_file_name}.conf")

        self._data = typedload.load(ConfigFactory.parse_file(config_file_path), self._data_structure)

        Logger().debug(f"{self.get_class_name} was created from {self._config_file_name}.conf file.")

    def get_data(self) -> Union[RowBlocksGroupingPipelineConfigData, ColumnsGroupingPipelineConfigData,
                                TransformationsExecutionerPipelineConfigData]:
        """
        Returns the config's named tuple.
        :return: Union[RowBlocksGroupingPipelineConfigData, ColumnsGroupingPipelineConfigData,
                                       TransformationsExecutionerPipelineConfigData]. Named tuple with data.
        """
        return self._data

    def get_data_as_dict(self) -> Dict[Any, Any]:
        """
        Gets the data (named tuple) into dictionary.
        :return: Dict[Any, Any].
        """
        return dict(self._data._asdict())
