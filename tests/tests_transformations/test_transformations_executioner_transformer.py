"""
Tester

NOTE: Proper values are not tested, only with inverse. The reason is that all the transformers are tested. But
      should be added for sure.
"""
from typing import List

import pytest
from numpy import array, testing
from pandas import DataFrame

from src.constants.global_constants import FP, P
from src.data.anomaly_detection_sample_data import AnomalyDetectionSampleData, ATTRS
from src.exceptions.data_exception import MismatchedDimension, IncorrectDataStructure
from src.exceptions.development_exception import NoProperOptionInIf, NotValidOperation
from src.transformations.transformations_executioner_transformer import TransformationsExecutionerTransformer, \
    TransformerConfiguration

C_DIFF = TransformerConfiguration(
    name="DifferenceTransformer", fit=True, params_for_fit={"periods": 1}, params_fitted={}
)
C_DIFF_PERC = TransformerConfiguration(
    name="DifferencePercentageChangeTransformer", fit=True, params_for_fit={"periods": 1}, params_fitted={}
)
C_LOG = TransformerConfiguration(
    name="LogTransformer", fit=True, params_for_fit={"base": 10.}, params_fitted={}
)
C_LOG_FITTED = TransformerConfiguration(
    name="LogTransformer", fit=False, params_for_fit={}, params_fitted={"base": 10.}
)
C_MM_SINGLE = TransformerConfiguration(
    name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": False}, params_fitted={}
)
C_MM_MULTI = TransformerConfiguration(
    name="MinMaxScalingTransformer", fit=True, params_for_fit={"single_columns": True}, params_fitted={}
)
C_Q = TransformerConfiguration(
    name="QuantileTransformer", fit=True, params_for_fit={}, params_fitted={}
)
C_SHIFT = TransformerConfiguration(
    name="ShiftByValueTransformer", fit=True, params_for_fit={"shift_value": -0.5}, params_fitted={}
)


def _do_transformation(type_of_method: str, X_df: DataFrame, attrs: List[str], \
                       configurations: List[TransformerConfiguration]) -> DataFrame:
    """
    Does the transformation.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param X_df: DataFrame. Data frame to be used for fitting.
    :param attrs: List[str]. List of attributes of the data frame to be transformed.
    :param configurations: List[TransformerConfiguration]. Configuration of the set of transformations to be
                           applied.
    :return: Tuple[DataFrame, MinMaxScalingTransformer].
    """
    transformer = TransformationsExecutionerTransformer()
    if type_of_method == FP:
        data_output = transformer.fit_predict(X_df, attrs, configurations)
    elif type_of_method == P:
        transformer.fit(X_df, attrs, configurations)
        data_output = transformer.predict(X_df)
    else:
        raise NoProperOptionInIf(transformer.get_class_info().class_name)
    return data_output, transformer


@pytest.mark.parametrize("type_of_method, attrs, configurations, abs_values, add_zeros, n, seed_number",
                         [
                             (FP, ATTRS, [C_MM_SINGLE, C_SHIFT], False, False, 20, 876),
                             (P, ATTRS, [C_MM_SINGLE, C_SHIFT], False, False, 20, 876),
                             (FP, ATTRS, [C_LOG, C_SHIFT], True, True, 25, 8769),
                             (P, ATTRS, [C_LOG, C_SHIFT], True, True, 25, 8769),
                             (FP, ATTRS, [C_LOG_FITTED, C_MM_MULTI, C_SHIFT], True, True, 25, 769),
                             (P, ATTRS, [C_LOG_FITTED, C_MM_MULTI, C_SHIFT], True, True, 25, 769)
                         ])
def test_inverse(type_of_method: str, attrs: List[str], configurations: List[TransformerConfiguration], \
                 abs_values: bool, add_zeros: bool, n: int, seed_number: int) -> None:
    """
    Tests that the values in the data frame are the same.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param attrs: List[str]. List of attributes of the data frame to be transformed.
    :param configurations: List[TransformerConfiguration]. Configuration of the set of transformations to be
                               applied.
    :param abs_values: bool. If generated values are absolute.
    :param add_zeros: bool. If to add zeros in the beginning - because of logarithic transformation.
    :param n: int. Number of observation.
    :param seed_number: Optional[int].
    """
    df = AnomalyDetectionSampleData(abs_values=abs_values, add_zeros=add_zeros).generate(n=n, seed_number=seed_number)

    output_data, transformer = _do_transformation(type_of_method, df, attrs, configurations)
    data_input_restored = transformer.inverse(output_data)

    # Round trip must recover the original input (predict/inverse no longer mutate in place).
    testing.assert_equal(df[attrs].to_numpy().round(4), data_input_restored[attrs].to_numpy().round(4))


@pytest.mark.parametrize("type_of_method, attrs, configurations, abs_values, add_zeros, n, seed_number",
                         [
                             (FP, ATTRS, [C_MM_SINGLE, C_SHIFT], False, False, 20, 876),
                             (P, ATTRS, [C_MM_SINGLE, C_SHIFT], False, False, 20, 876),
                             (FP, ATTRS, [C_LOG, C_SHIFT], True, True, 25, 8769),
                             (P, ATTRS, [C_LOG, C_SHIFT], True, True, 25, 8769),
                             (FP, ATTRS, [C_LOG, C_MM_MULTI, C_SHIFT], True, True, 25, 769),
                             (P, ATTRS, [C_LOG, C_MM_MULTI, C_SHIFT], True, True, 25, 769)
                         ])
def test_restoration(type_of_method: str, attrs: List[str], configurations: List[TransformerConfiguration], \
                     abs_values: bool, add_zeros: bool, n: int, seed_number: int) -> None:
    """
    Tests that the values in the data frame are the same.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param attrs: List[str]. List of attributes of the data frame to be transformed.
    :param configurations: List[TransformerConfiguration]. Configuration of the set of transformations to be
                               applied.
    :param abs_values: bool. If generated values are absolute.
    :param add_zeros: bool. If to add zeros in the beginning - because of logarithic transformation.
    :param n: int. Number of observation.
    :param seed_number: Optional[int].
    """
    df = AnomalyDetectionSampleData(abs_values=abs_values, add_zeros=add_zeros).generate(n=n, seed_number=seed_number)
    output_data, transformer = _do_transformation(type_of_method, df.copy(), attrs, configurations)

    transformer_restored = TransformationsExecutionerTransformer()
    transformer_restored.restore_from_params(transformer.get_params())
    output_data_restored = transformer_restored.predict(df)

    testing.assert_equal(output_data[attrs].to_numpy().round(4), output_data_restored[attrs].to_numpy().round(4))


@pytest.mark.parametrize("type_of_method, attrs, configurations, abs_values, add_zeros, n, seed_number",
                         [
                             (FP, [ATTRS[0]], [C_DIFF], False, False, 25, 79),
                             (P, [ATTRS[0]], [C_DIFF], False, False, 25, 79),
                             (FP, [ATTRS[0]], [C_DIFF_PERC], False, False, 25, 179),
                             (P, [ATTRS[0]], [C_DIFF_PERC], False, False, 25, 179),
                             (FP, [ATTRS[0]], [C_Q], False, False, 25, 77),
                             (P, [ATTRS[0]], [C_Q], False, False, 25, 77),
                             (FP, [ATTRS[0]], [C_Q, C_MM_SINGLE], False, False, 25, 77),
                             (P, [ATTRS[0]], [C_Q, C_MM_SINGLE], False, False, 25, 77)
                         ])
def test_no_invertible(type_of_method: str, attrs: List[str], configurations: List[TransformerConfiguration], \
                       abs_values: bool, add_zeros: bool, n: int, seed_number: int) -> None:
    """
    Tests that the values in the data frame are the same.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param attrs: List[str]. List of attributes of the data frame to be transformed.
    :param configurations: List[TransformerConfiguration]. Configuration of the set of transformations to be
                               applied.
    :param abs_values: bool. If generated values are absolute.
    :param add_zeros: bool. If to add zeros in the beginning - because of logarithic transformation.
    :param n: int. Number of observation.
    :param seed_number: Optional[int].
    """
    df = AnomalyDetectionSampleData(abs_values=abs_values, add_zeros=add_zeros).generate(n=n, seed_number=seed_number)
    _, t = _do_transformation(type_of_method, df, attrs, configurations)
    with pytest.raises(NotValidOperation):
        t.inverse(df)


@pytest.mark.parametrize("type_of_method, attrs, configurations, abs_values, add_zeros, n, seed_number",
                         [
                             (FP, ATTRS, [C_Q], False, False, 25, 769),
                             (P, ATTRS, [C_Q], False, False, 25, 769),
                             (FP, ATTRS, [C_DIFF], False, False, 25, 69),
                             (P, ATTRS, [C_DIFF], False, False, 25, 69),
                             (FP, ATTRS, [C_Q, C_MM_SINGLE, C_SHIFT], False, False, 25, 9),
                             (P, ATTRS, [C_Q, C_MM_SINGLE, C_SHIFT], False, False, 25, 9)
                         ])
def test_dimension_missmatch(type_of_method: str, attrs: List[str], configurations: List[TransformerConfiguration], \
                             abs_values: bool, add_zeros: bool, n: int, seed_number: int) -> None:
    """
    Tests that the values in the data frame are the same.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param attrs: List[str]. List of attributes of the data frame to be transformed.
    :param configurations: List[TransformerConfiguration]. Configuration of the set of transformations to be
                               applied.
    :param abs_values: bool. If generated values are absolute.
    :param add_zeros: bool. If to add zeros in the beginning - because of logarithic transformation.
    :param n: int. Number of observation.
    :param seed_number: Optional[int].
    """
    df = AnomalyDetectionSampleData(abs_values=abs_values, add_zeros=add_zeros).generate(n=n, seed_number=seed_number)
    with pytest.raises(MismatchedDimension):
        _, _ = _do_transformation(type_of_method, df, attrs, configurations)


@pytest.mark.parametrize("type_of_method, attrs, configurations, abs_values, add_zeros, n, seed_number",
                         [
                             (FP, ATTRS, [], False, False, 25, 769),
                             (P, ATTRS, [], False, False, 25, 769)
                         ])
def test_empty_config(type_of_method: str, attrs: List[str], configurations: List[TransformerConfiguration], \
                      abs_values: bool, add_zeros: bool, n: int, seed_number: int) -> None:
    """
    Tests that the values in the data frame are the same.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    :param attrs: List[str]. List of attributes of the data frame to be transformed.
    :param configurations: List[TransformerConfiguration]. Configuration of the set of transformations to be
                               applied.
    :param abs_values: bool. If generated values are absolute.
    :param add_zeros: bool. If to add zeros in the beginning - because of logarithic transformation.
    :param n: int. Number of observation.
    :param seed_number: Optional[int].
    """
    df = AnomalyDetectionSampleData(abs_values=abs_values, add_zeros=add_zeros).generate(n=n, seed_number=seed_number)

    output_data, _ = _do_transformation(type_of_method, df, attrs, configurations)

    testing.assert_equal(output_data[attrs].to_numpy().round(4), df[attrs].to_numpy().round(4))


@pytest.mark.parametrize("type_of_method",
                         [
                             (FP),
                             (P)
                         ])
def test_fit_from_fitted_config(type_of_method: str) -> None:
    """
    Tests fitting from None data frame.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    """
    df = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).generate(n=25, seed_number=876)
    output_data_fit, _ = _do_transformation(type_of_method, df, ATTRS, [C_LOG])

    transformer = TransformationsExecutionerTransformer()
    transformer.fit(None, ATTRS, [C_LOG_FITTED])
    output_data_from_fitted = transformer.predict(df)

    testing.assert_equal(output_data_fit[ATTRS].to_numpy().round(4), output_data_from_fitted[ATTRS].to_numpy().round(4))


@pytest.mark.parametrize("type_of_method",
                         [
                             (FP),
                             (P)
                         ])
def test_error_in_none_data_frame(type_of_method: str) -> None:
    """
    Tests fitting from None data frame.
    :param type_of_method: str. Transformation type ("f" for fit, "p" for predict, "fp" for fit_predict).
    """
    with pytest.raises(IncorrectDataStructure):
        _, _ = _do_transformation(type_of_method, None, ATTRS, [C_LOG])


# ---------------------------------------------------------------------------
# Value correctness: assert actual transformed values, not just round-trips
# ---------------------------------------------------------------------------

def test_value_correctness_shift() -> None:
    """
    Tests that ShiftByValueTransformer adds the constant to every value.
    """
    df = DataFrame({"a": [1., 2., 3., 4.]})
    c_shift = TransformerConfiguration(
        name="ShiftByValueTransformer", fit=True, params_for_fit={"shift_value": 5.}, params_fitted={}
    )
    output = TransformationsExecutionerTransformer().fit_predict(df, ["a"], [c_shift])

    testing.assert_equal(output["a"].to_numpy(), array([6., 7., 8., 9.]))


def test_value_correctness_log() -> None:
    """
    Tests that LogTransformer with base 10 maps x to log10(x + 1) (the +1 makes it zero-safe),
    so inputs of (power of ten minus one) map to clean exponents.
    """
    df = DataFrame({"a": [9., 99., 999., 9999.]})
    output = TransformationsExecutionerTransformer().fit_predict(df, ["a"], [C_LOG])

    testing.assert_allclose(output["a"].to_numpy(), array([1., 2., 3., 4.]), atol=1e-9)


# ---------------------------------------------------------------------------
# QuantileTransformer serialization: state must survive get_params/restore
# ---------------------------------------------------------------------------

C_QUANT = TransformerConfiguration(
    name="QuantileTransformer", fit=True, params_for_fit={}, params_fitted={}
)


def test_quantile_serialization_round_trip() -> None:
    """
    Tests that a QuantileTransformer-containing config can be serialized via get_params and
    restored into a fresh executioner that produces identical predictions. This exercises the
    QuantileTransformer.get_params/restore_from_params overrides.
    """
    df = AnomalyDetectionSampleData(abs_values=False, add_zeros=False).generate(n=20, seed_number=321)
    attrs = [ATTRS[0]]  # QuantileTransformer is one-dimensional

    transformer = TransformationsExecutionerTransformer()
    output = transformer.fit_predict(df.copy(), attrs, [C_QUANT])

    transformer_restored = TransformationsExecutionerTransformer()
    transformer_restored.restore_from_params(transformer.get_params())
    output_restored = transformer_restored.predict(df.copy())

    testing.assert_equal(output[attrs].to_numpy().round(4), output_restored[attrs].to_numpy().round(4))


# ---------------------------------------------------------------------------
# Stale config restoration: fit=True with populated params_fitted and no data
# ---------------------------------------------------------------------------

# Simulates a stale config exported before the fit=False fix: fit is still True
# but params_fitted is populated.
C_LOG_STALE = TransformerConfiguration(
    name="LogTransformer", fit=True, params_for_fit={"base": 10.}, params_fitted={"base": 10.}
)


@pytest.mark.parametrize("type_of_method",
                         [
                             (FP),
                             (P)
                         ])
def test_stale_config_restores_gracefully(type_of_method: str) -> None:
    """
    Tests that a stale configuration (fit=True but params_fitted populated) can be
    restored from None data without raising IncorrectDataStructure.
    """
    df = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).generate(n=25, seed_number=876)
    # First: produce reference output by fitting normally
    output_data_fit, _ = _do_transformation(type_of_method, df, ATTRS, [C_LOG])

    # Second: restore from stale config (fit=True, params populated, data=None)
    transformer = TransformationsExecutionerTransformer()
    transformer.fit(None, ATTRS, [C_LOG_STALE])
    output_data_stale = transformer.predict(df)

    testing.assert_equal(
        output_data_fit[ATTRS].to_numpy().round(4),
        output_data_stale[ATTRS].to_numpy().round(4)
    )


def test_stale_config_flips_fit_to_false() -> None:
    """
    Tests that after restoring from a stale config, the fit flag is flipped to False
    in the stored configurations.
    """
    transformer = TransformationsExecutionerTransformer()
    transformer.fit(None, ATTRS, [C_LOG_STALE])

    params = transformer.get_params()
    for conf in params["configurations"]:
        assert conf["fit"] is False, (
            f"Expected fit=False after stale restore, got fit={conf['fit']} "
            f"for {conf['name']}"
        )


def test_stale_config_minmax_restores_and_matches() -> None:
    """
    Tests the stale config path with MinMaxScalingTransformer: fit the pipeline on data,
    then simulate a stale config by keeping fit=True alongside the populated params_fitted,
    and verify the restored transformer produces identical output.
    """
    df = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).generate(n=30, seed_number=42)

    # Fit normally to get reference output and fitted params
    transformer_orig = TransformationsExecutionerTransformer()
    output_orig = transformer_orig.fit_predict(df.copy(), ATTRS, [C_LOG, C_MM_MULTI])
    fitted_params = transformer_orig.get_params()

    # Build stale configs: fit=True but params_fitted populated from the fitted transformer
    stale_configurations = [
        TransformerConfiguration(
            name=conf["name"], fit=True,
            params_for_fit=conf["params_for_fit"], params_fitted=conf["params_fitted"]
        )
        for conf in fitted_params["configurations"]
    ]

    # Restore from stale config with no data
    transformer_stale = TransformationsExecutionerTransformer()
    transformer_stale.fit(None, ATTRS, stale_configurations)
    output_stale = transformer_stale.predict(df.copy())

    testing.assert_equal(
        output_orig[ATTRS].to_numpy().round(4),
        output_stale[ATTRS].to_numpy().round(4)
    )


def test_stale_config_chain_inverse_matches() -> None:
    """
    Tests that a pipeline restored from stale configs (fit=True + populated params_fitted)
    produces the same inverse result as the original fitted pipeline.
    """
    df = AnomalyDetectionSampleData(abs_values=True, add_zeros=True).generate(n=30, seed_number=99)

    # Fit normally
    transformer_orig = TransformationsExecutionerTransformer()
    output_orig = transformer_orig.fit_predict(df.copy(), ATTRS, [C_LOG, C_MM_MULTI, C_SHIFT])
    inverse_orig = transformer_orig.inverse(output_orig.copy())
    fitted_params = transformer_orig.get_params()

    # Build stale configs
    stale_configurations = [
        TransformerConfiguration(
            name=conf["name"], fit=True,
            params_for_fit=conf["params_for_fit"], params_fitted=conf["params_fitted"]
        )
        for conf in fitted_params["configurations"]
    ]

    # Restore from stale config
    transformer_stale = TransformationsExecutionerTransformer()
    transformer_stale.fit(None, ATTRS, stale_configurations)
    output_stale = transformer_stale.predict(df.copy())
    inverse_stale = transformer_stale.inverse(output_stale.copy())

    testing.assert_equal(
        inverse_orig[ATTRS].to_numpy().round(4),
        inverse_stale[ATTRS].to_numpy().round(4)
    )
