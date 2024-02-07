"""
Pipeline
"""
from typing import Optional, List

from pandas import DataFrame

from src.pipelines.base_pipeline import BasePipeline
from src.pipelines.columns_grouping_pipeline_config import ColumnsGroupingPipelineConfig
from src.transformations.group_df_columns_transformer import GroupDFColumnsTransformer


class ColumnsGroupingPipeline(BasePipeline):  # type: ignore
    """
    Pipeline for doing grouping operations on the columns of a data frame. Other dimension is just a row.
    """

    def __init__(self, config_file_name: Optional[str] = None) -> None:
        BasePipeline.__init__(self, class_name="",
                              config_file_name=config_file_name,
                              config_class=ColumnsGroupingPipelineConfig)

        self._column_transformer = GroupDFColumnsTransformer()

    # pylint: disable=arguments-differ
    def execute(self, dfs: List[DataFrame]) -> List[DataFrame]:
        """
        Executes operations equivalent to config file for one data frame.
        Assumes all the data frames have enough observation for the necessary transformations.
        :param dfs: List[Dataframe]. List of data frames to be updated.
        :return: List[DataFrame].
        """
        dfs_out = []
        for df in dfs:
            dfs_out.append(self._execute_one_data_frame(df))
        return dfs_out

    # pylint: enable=arguments-differ
    def _execute_one_data_frame(self, df: DataFrame) -> DataFrame:
        """
        Executes operations equivalent to config file for one data frame.
        :param df: DataFrame. Data frame to be transformed.
        :return: DataFrame. Original data frame with new attributes from configuration.
        """
        for grp in self._config_data.columns_grouping:
            if grp.create:
                df[grp.new_attr_name] = self._column_transformer.fit_predict(
                    df, grp.attrs, grp.grouping_fun, grp.params
                )

        return df
