"""
Parent class for all pipelines.

Pipelines are high level operations with its own configs.
"""

from abc import abstractmethod
from typing import Union, Optional, List

from pandas import DataFrame

from src.pipelines.columns_grouping_pipeline_config import ColumnsGroupingPipelineConfig
from src.pipelines.columns_grouping_pipeline_config_data import ColumnsGroupingPipelineConfigData
from src.pipelines.new_attributes_pipeline_config import NewAttributesPipelineConfig
from src.pipelines.new_attributes_pipeline_config_data import NewAttributesPipelineConfigData
from src.pipelines.post_processing_pipeline_config import PostProcessingPipelineConfig
from src.pipelines.post_processing_pipeline_config_data import PostProcessingPipelineConfigData
from src.pipelines.row_blocks_grouping_pipeline_config import RowBlocksGroupingPipelineConfig
from src.pipelines.row_blocks_grouping_pipeline_config_data import RowBlocksGroupingPipelineConfigData
from src.pipelines.transfomations_executioner_pipeline_config import TransformationsExecutionerPipelineConfig
from src.pipelines.transfomations_executioner_pipeline_config_data import TransformationsExecutionerPipelineConfigData
from src.utils.meta_class import MetaClass, PIPELINE_TYPE_NAME


class BasePipeline(MetaClass):  # type:ignore
    """
    Parent/base class for all pipelines to ensure the same interface.
    """

    def __init__(self, class_name: str, config_file_name: Optional[str], \
                 config_class: Union[TransformationsExecutionerPipelineConfig, NewAttributesPipelineConfig,
                 RowBlocksGroupingPipelineConfig, ColumnsGroupingPipelineConfig,
                 PostProcessingPipelineConfig]) -> None:
        MetaClass.__init__(self, class_type=PIPELINE_TYPE_NAME, class_name=class_name)

        self._config_data: Union[
            NewAttributesPipelineConfigData,
            TransformationsExecutionerPipelineConfigData,
            RowBlocksGroupingPipelineConfigData,
            ColumnsGroupingPipelineConfigData,
            PostProcessingPipelineConfigData
        ]
        self.read_config_data(config_file_name, config_class)

    @abstractmethod
    def _execute_one_data_frame(self, df: DataFrame) -> DataFrame:
        """
        Executes operations equivalent to config file for one data frame.
        :param df: DataFrame. Data frame to be transformed.
        :return: DataFrame. Original data frame with new attributes from configuration.
        """

    def execute(self, dfs: List[DataFrame]) -> List[DataFrame]:
        """
        Executes operations equivalent to config file.
        Assumes all the data frames have enough observation for the necessary transformations.
        :param dfs: List[Dataframe]. List of data frames to be updated.
        :return: List[DataFrame].
        """
        dfs_out = []
        for df in dfs:
            dfs_out.append(self._execute_one_data_frame(df))
        return dfs_out

    def read_config_data(self, config_file_name: Optional[str], config_class: \
            Union[TransformationsExecutionerPipelineConfig, NewAttributesPipelineConfig,
            RowBlocksGroupingPipelineConfig, ColumnsGroupingPipelineConfig, PostProcessingPipelineConfig]) \
            -> None:
        """
        Reads the config data.
        :param config_file_name: Optional[str].
        :param config_class: Union[TransformationsExecutionerPipelineConfig, NewAttributesPipelineConfig,
                                   RowBlocksGroupingPipelineConfig, ColumnsGroupingPipelineConfig,
                                   PostProcessingPipelineConfig].
        """
        if config_file_name is not None:
            self._config_data = config_class(config_file_name).get_data()

    def set_config_data(self, config_data: Union[NewAttributesPipelineConfigData,
                              TransformationsExecutionerPipelineConfigData,
                              RowBlocksGroupingPipelineConfigData,
                              ColumnsGroupingPipelineConfigData,
                              PostProcessingPipelineConfigData]) -> None:
        """
        Sets the config data.
        :param config_data: Union[NewAttributesPipelineConfigData, TransformationsExecutionerPipelineConfigData,
                                  RowBlocksGroupingPipelineConfigData, ColumnsGroupingPipelineConfigData,
                                  PostProcessingPipelineConfigData].
        """
        self._config_data = config_data

    def get_config_data(self) -> Union[NewAttributesPipelineConfigData, TransformationsExecutionerPipelineConfigData,
                                       RowBlocksGroupingPipelineConfigData, ColumnsGroupingPipelineConfigData,
                                       PostProcessingPipelineConfigData]:
        """
        Gets the config data.
        :return: Union[NewAttributesPipelineConfigData, TransformationsExecutionerPipelineConfigData,
                       RowBlocksGroupingPipelineConfigData, ColumnsGroupingPipelineConfigData,
                       PostProcessingPipelineConfigData].
        """
        return self._config_data
