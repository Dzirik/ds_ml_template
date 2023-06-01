"""
Pipelines configuration file.
"""

from pprint import pprint

from src.pipelines.columns_grouping_pipeline_config_data import ColumnsGroupingPipelineConfigData
from src.utils.base_config import BaseConfig


class ColumnsGroupingPipelineConfig(BaseConfig):  # type:ignore
    """
    Pipelines configuration file.
    """

    def __init__(self, config_file_name: str) -> None:
        """
        :param config_file_name: str. Name of the file in configuration folder without that .conf part.
        """
        BaseConfig.__init__(
            self,
            class_name="ColumnsGroupingPipelineConfig",
            config_file_name=config_file_name,
            data_structure=ColumnsGroupingPipelineConfigData
        )


if __name__ == "__main__":
    CONFIG_FILE_NAME = "pipeline_columns_grouping_documentation"

    config_data = ColumnsGroupingPipelineConfig(CONFIG_FILE_NAME).get_data()

    print("\n")
    pprint(config_data.name)
    print("\n")
    pprint(config_data.columns_grouping)
