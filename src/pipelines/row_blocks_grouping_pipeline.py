"""
Pipeline
"""
from typing import Optional, List, Dict

from pandas import DataFrame

from src.constants.global_constants import ATTR_TIME
from src.pipelines.base_pipeline import BasePipeline
from src.pipelines.row_blocks_grouping_pipeline_config import RowBlocksGroupingPipelineConfig
from src.transformations.group_df_blocks_by_time_transformer import GroupDFBlocksByTimeTransformer


class RowBlocksGroupingPipeline(BasePipeline):  # type: ignore
    """
    Pipeline for doing grouping operations on the block of time (rows) data of a data frame.
    """

    def __init__(self, config_file_name: Optional[str] = None) -> None:
        BasePipeline.__init__(self, class_name="",
                              config_file_name=config_file_name,
                              config_class=RowBlocksGroupingPipelineConfig)

        self._block_transformer = GroupDFBlocksByTimeTransformer()

    # pylint: disable=arguments-differ
    def execute(self, df: DataFrame) -> DataFrame:
        """
        Transforms row blocks. Please see fe notebook or transformer documentation for more information.
        :param df: DataFrame. Data frame to be transformed.
        :return: DataFrame. Data frame with DATETIME attribute and attributes from grouping configuration.
        """
        data_frames = []

        for grp in self._config_data.row_blocks_grouping:
            if grp.create:
                data_frames.append(self._create_grouped_attribute(df, grp.attrs, grp.fun, grp.rename))
        if len(data_frames) == 0:
            return df
        return self._merge_data_frames_horizontally(data_frames)

    # pylint: enable=arguments-differ

    def _create_grouped_attribute(self, df: DataFrame, attrs: List[str], grp_fun: str,
                                  rename_dict: Optional[Dict[str, str]] = None) -> DataFrame:
        """
        Creates grouped data frame from data frame df.
        :param df: DataFrame.
        :param attrs: List[str]. List of parameters to be grouped. Datetime has to be included as first one.
        :param grp_fun: str. Grouping function from the GroupDFBlocksByTimeTransformer.
        :param rename_dict: Optional[Dict[str, str]]. Dictionary for renaming the column. If None, nothing is renamed.
        :return: DataFrame. Grouped data frame.
        """
        df_grp = self._block_transformer.fit_predict(df[attrs], ATTR_TIME,
                                                     self._config_data.grouping_window_len, grp_fun)
        df_grp.index = df_grp[ATTR_TIME].values
        df_grp.index.name = ""
        if rename_dict is not None:
            df_grp.rename(columns=rename_dict, inplace=True)
        return df_grp

    @staticmethod
    def _merge_data_frames_horizontally(data_frames: List[DataFrame]) -> DataFrame:
        """
        Merges data frames in dataframes list with df_dt data frame in horizontal matter. The indices should be the
        same, the shape as well. The testing of identical indices is done.
        :param data_frames: List[DataFrame]. List of data frames.
        :return: DataFrame. Merged data frame.
        """
        df_out = data_frames[0][[ATTR_TIME]]
        for df in data_frames:
            if df[[ATTR_TIME]].equals(df_out[[ATTR_TIME]]):
                attr_name = list(df.columns)
                attr_name.remove(ATTR_TIME)
                attr_name = attr_name[0]
                df_out[attr_name] = df[attr_name]

        return df_out
