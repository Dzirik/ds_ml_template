"""
Pipeline
"""
from typing import Optional

from pandas import DataFrame

from src.exceptions.development_exception import NoProperOptionInIf
from src.pipelines.base_pipeline import BasePipeline
from src.pipelines.post_processing_pipeline_config import PostProcessingPipelineConfig


class PostProcessingPipeline(BasePipeline):  # type: ignore
    """
    Pipeline for doing grouping operations on the block of time (rows) data of a data frame.
    """

    def __init__(self, config_file_name: Optional[str] = None) -> None:
        BasePipeline.__init__(self, class_name="PostProcessingPipeline",
                              config_file_name=config_file_name,
                              config_class=PostProcessingPipelineConfig)

    # pylint: disable=arguments-differ
    def execute(self, df: DataFrame) -> DataFrame:
        """
        Transforms row blocks. Please see fe notebook or transformer documentation for more information.
        :param df: DataFrame. Data frame to be transformed.
        :return: DataFrame. Data frame with DATETIME attribute and attributes from grouping configuration.
        """
        for conf in self._config_data.post_processing_operations:
            if conf.create:
                if conf.name_of_operation == "remove_from_head":
                    n = int(conf.parameters["n_of_rows"])
                    df = df.iloc[n:]
                elif conf.name_of_operation == "remove_from_tail":
                    n = int(conf.parameters["n_of_rows"])
                    df = df.iloc[:-n]
                else:
                    print(conf.name_of_operation)
                    raise NoProperOptionInIf
        return df

    # pylint: enable=arguments-differ
