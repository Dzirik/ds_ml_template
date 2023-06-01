"""
Config for pre model pipeline.
"""

from pprint import pprint

from src.pipelines.transfomations_executioner_pipeline_config_data import TransformationsExecutionerPipelineConfigData
from src.utils.base_config import BaseConfig


class TransformationsExecutionerPipelineConfig(BaseConfig):  # type:ignore
    """
    Class for handling config file for pre model pipeline.
    """

    def __init__(self, config_file_name: str) -> None:
        """
        :param config_file_name: str. Name of the file in configuration folder without that .conf part.
        """
        BaseConfig.__init__(
            self,
            class_name="PreModelPipelineConfig",
            config_file_name=config_file_name,
            data_structure=TransformationsExecutionerPipelineConfigData
        )


if __name__ == "__main__":
    CONFIG_FILE_NAME = "pipeline_transformations_executioner_x_documentation"

    tr_exec_data = TransformationsExecutionerPipelineConfig(CONFIG_FILE_NAME).get_data()

    pprint(tr_exec_data)
    print("\n")
