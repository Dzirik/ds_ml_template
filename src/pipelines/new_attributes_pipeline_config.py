"""
Pipelines configuration file.
"""

from pprint import pprint

from src.pipelines.new_attributes_pipeline_config_data import NewAttributesPipelineConfigData
from src.utils.base_config import BaseConfig


class NewAttributesPipelineConfig(BaseConfig):  # type: ignore
    """
    Pipelines configuration file.
    """

    def __init__(self, config_file_name: str) -> None:
        """
        :param config_file_name: str. Name of the file in configuration folder without that .conf part.
        """
        BaseConfig.__init__(
            self,
            class_name="NewAttributesPipelineConfig",
            config_file_name=config_file_name,
            data_structure=NewAttributesPipelineConfigData
        )


if __name__ == "__main__":
    CONFIG_FILE_NAME = "pipeline_new_attributes_documentation"

    config_data = NewAttributesPipelineConfig(CONFIG_FILE_NAME).get_data()

    print("\n")
    pprint(config_data.name)
    print("\n")
    pprint(config_data.new_attributes)
