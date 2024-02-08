"""
Pipeline

See documentation notebooks for more details.
"""
from typing import List, Tuple, Optional

from pandas import DataFrame, concat

from src.constants.global_constants import FP, P, I
from src.data.saver_and_loader import SaverAndLoader
from src.exceptions.development_exception import NoProperOptionInIf, NotReadyFunctionality
from src.pipelines.base_pipeline import BasePipeline
from src.pipelines.transfomations_executioner_pipeline_config import TransformationsExecutionerPipelineConfig
from src.transformations.transformations_executioner_transformer import TransformationsExecutionerTransformer


class TransformationsExecutionerlPipeline(BasePipeline):  # type:ignore
    """
    Pipeline for pre-model transformations and their inverses.
    """

    def __init__(self, config_file_name: Optional[str] = None) -> None:
        BasePipeline.__init__(self, class_name="YDataCreationPipeline",
                              config_file_name=config_file_name,
                              config_class=TransformationsExecutionerPipelineConfig)

        self._exec_trs: List[TransformationsExecutionerTransformer]
        self._saver_and_loader = SaverAndLoader()

    def fit(self, dfs: List[Optional[DataFrame]]) -> None:
        """
        Fits the transformation's executioner.
        :param dfs: List[Optional[DataFrame]]. List of training data frames. In just one element and equals None,
        then transformers are restored from fitted params.
        """
        if len(dfs) == 1 and dfs[0] is None:
            df_concatenated = None
        else:
            df_concatenated = self._concatenate_data_frames(dfs, axis=0)

        self._exec_trs = []
        for exec_conf in self._config_data.trans_executioners_confs:
            if exec_conf.create:
                exec_tr = TransformationsExecutionerTransformer()
                exec_tr.fit(
                    data=df_concatenated,
                    attrs=exec_conf.attrs,
                    configurations=exec_conf.configurations
                )
                self._exec_trs.append(exec_tr)

    def _execute_one_data_frame(self, df: DataFrame) -> DataFrame:
        """
        Executes operations equivalent to config file for one data frame.
        IT IS NOT NECESSARY HERE, NOT VERY CONCEPTUAL, BUT ...
        :param df: DataFrame. Data frame to be transformed.
        :return: DataFrame. Original data frame with new attributes from configuration.
        """
        return DataFrame()

    def predict(self, dfs: List[DataFrame]) -> List[DataFrame]:
        """
        Predicts.
        :param dfs: List[DataFrame].
        :return: List[DataFrame].
        """
        dfs_out = []
        for df in dfs:
            df_tr = df.copy()
            for exec_tr in self._exec_trs:
                df_tr = exec_tr.predict(df_tr)
            dfs_out.append(df_tr)
        return dfs_out

    def fit_predict(self, dfs: List[DataFrame]) -> List[DataFrame]:
        """
        Fits and predicts.
        :param dfs: List[DataFrame].
        :return: List[DataFrame].
        """
        self.fit(dfs)
        return self.predict(dfs)

    def inverse(self, dfs: List[DataFrame]) -> List[DataFrame]:
        """
        Does the inverse.
        :param dfs: List[DataFrame].
        :return: List[DataFrame].
        """
        # TODO: Do inverse.

    # pylint: disable=arguments-differ
    def execute(self, dfs_train: Optional[List[DataFrame]], dfs_test: Optional[List[DataFrame]], type_of_method: str) \
            -> Tuple[Optional[List[DataFrame]], Optional[List[DataFrame]]]:
        """
        Executes the pipeline.
        :param dfs_train: Optional[List[DataFrame]].
        :param dfs_test: Optional[List[DataFrame]].
        :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
        :return: Tuple[Optional[List[DataFrame]], Optional[List[DataFrame]]]. dfs_train_out, dfs_test_out.
        """
        dfs_train_out = None
        dfs_test_out = None
        if type_of_method == FP:
            if dfs_train is not None:
                self.fit(dfs_train)
                dfs_train_out = self.predict(dfs_train)
                if dfs_test is not None:
                    dfs_test_out = self.predict(dfs_test)
        elif type_of_method == P:
            # in this case the fitted parameters are there, so just transformers needs to be restored.
            self.fit([None])
            if dfs_test is not None:
                dfs_test_out = self.predict(dfs_test)
        elif type_of_method == I:
            # TODO: Add inverse.
            raise NotReadyFunctionality
        else:
            raise NoProperOptionInIf
        return dfs_train_out, dfs_test_out

    @staticmethod
    def _concatenate_data_frames(dfs: List[DataFrame], axis: int = 0) -> DataFrame:
        """
        Concatenates data frames into one.
        :param dfs: List[DataFrame]. List of data frames.
        :param axis: int. Axis around which it should be concatenated.
        """
        if len(dfs) == 1:
            return dfs[0].copy()

        df_concatenated = concat([df.copy() for df in dfs], axis=axis)

        assert df_concatenated.shape[0] == sum([df.shape[0] for df in dfs])

        return df_concatenated

    # pylint: enable=arguments-differ
