"""
Pipelines configuration file.
"""

from pprint import pprint

from src.pipelines.post_processing_pipeline_config_data import PostProcessingPipelineConfigData
from src.utils.base_config import BaseConfig


class PostProcessingPipelineConfig(BaseConfig):  # type: ignore
    """
    Pipelines configuration file.
    """

    def __init__(self, config_file_name: str) -> None:
        """
        :param config_file_name: str. Name of the file in configuration folder without that .conf part.
        """
        BaseConfig.__init__(
            self,
            class_name="PostProcessingPipelineConfig",
            config_file_name=config_file_name,
            data_structure=PostProcessingPipelineConfigData
        )


if __name__ == "__main__":
    CONFIG_FILE_NAME = "pipeline_post_processing_documentation"

    config_data = PostProcessingPipelineConfig(CONFIG_FILE_NAME).get_data()

    print("\n")
    pprint(config_data.name)
    print("\n")
    pprint(config_data.post_processing_operations)
