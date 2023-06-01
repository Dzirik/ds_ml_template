"""
Pipelines configuration file.
"""

from pprint import pprint

from src.pipelines.row_blocks_grouping_pipeline_config_data import RowBlocksGroupingPipelineConfigData
from src.utils.base_config import BaseConfig


class RowBlocksGroupingPipelineConfig(BaseConfig):  # type:ignore
    """
    Pipelines configuration file.
    """

    def __init__(self, config_file_name: str) -> None:
        """
        :param config_file_name: str. Name of the file in configuration folder without that .conf part.
        """
        BaseConfig.__init__(
            self,
            class_name="RowBlocksGroupingPipelineConfig",
            config_file_name=config_file_name,
            data_structure=RowBlocksGroupingPipelineConfigData
        )


if __name__ == "__main__":
    CONFIG_FILE_NAME = "pipeline_row_blocks_grouping_documentation"

    config_data = RowBlocksGroupingPipelineConfig(CONFIG_FILE_NAME).get_data()

    print("\n")
    pprint(config_data.name)
    print("\n")
    pprint(config_data.grouping_window_len)
    print("\n")
    pprint(config_data.row_blocks_grouping)
