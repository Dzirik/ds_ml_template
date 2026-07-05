"""
Tester for the transformations executioner pipeline, focused on the inverse path.
"""
from typing import List

from numpy import testing

from src.constants.global_constants import I
from src.data.anomaly_detection_sample_data import AnomalyDetectionSampleData, ATTRS
from src.pipelines.transfomations_executioner_pipeline import TransformationsExecutionerlPipeline
from src.pipelines.transfomations_executioner_pipeline_config_data import (
    TransformationExecutionerConfiguration,
    TransformationsExecutionerPipelineConfigData,
)
from src.transformations.transformations_executioner_transformer import (
    TransformationsExecutionerTransformer,
    TransformerConfiguration,
)

C_LOG = TransformerConfiguration(
    name="LogTransformer", fit=True, params_for_fit={"base": 10.}, params_fitted={}
)
C_MM_MULTI = TransformerConfiguration(
    name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": True}, params_fitted={}
)
C_SHIFT = TransformerConfiguration(
    name="ShiftByValueTransformer", fit=True, params_for_fit={"shift_value": -0.5}, params_fitted={}
)


def _config_data(configurations: List[TransformerConfiguration]) -> TransformationsExecutionerPipelineConfigData:
    """
    Builds a single-executioner pipeline config data over ATTRS.
    :param configurations: List[TransformerConfiguration].
    :return: TransformationsExecutionerPipelineConfigData.
    """
    return TransformationsExecutionerPipelineConfigData(
        name="test",
        trans_executioners_confs=[
            TransformationExecutionerConfiguration(create=True, attrs=ATTRS, configurations=configurations)
        ]
    )


def test_pipeline_inverse_round_trip() -> None:
    """
    Tests that pipeline.inverse recovers the original input after pipeline.predict
    for an all-invertible chain.
    """
    df = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).generate(n=25, seed_number=11)

    pipeline = TransformationsExecutionerlPipeline()
    pipeline.set_config_data(_config_data([C_LOG, C_MM_MULTI, C_SHIFT]))
    pipeline.fit([df.copy()])

    transformed = pipeline.predict([df.copy()])
    restored = pipeline.inverse(transformed)

    testing.assert_equal(df[ATTRS].to_numpy().round(4), restored[0][ATTRS].to_numpy().round(4))


def test_pipeline_execute_inverse() -> None:
    """
    Tests the execute() inverse branch (type_of_method == I): transformers are restored from
    fitted configs and the transformed test data is inverted back to the original input.
    """
    df = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).generate(n=25, seed_number=11)

    # Fit a helper to obtain configurations carrying fitted params (fit flipped to False).
    helper = TransformationsExecutionerTransformer()
    helper.fit(df.copy(), ATTRS, [C_LOG, C_MM_MULTI, C_SHIFT])
    fitted_confs = [TransformerConfiguration(**d) for d in helper.get_params()["configurations"]]
    transformed_df = helper.predict(df.copy())

    pipeline = TransformationsExecutionerlPipeline()
    pipeline.set_config_data(_config_data(fitted_confs))

    _, inverted = pipeline.execute(None, [transformed_df], I)

    assert inverted is not None
    testing.assert_equal(df[ATTRS].to_numpy().round(4), inverted[0][ATTRS].to_numpy().round(4))
