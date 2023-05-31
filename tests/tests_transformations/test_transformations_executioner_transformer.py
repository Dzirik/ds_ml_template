"""
Tester

NOTE: Proper values are not tested, only with inverse. The reason is that all the transformers are tested. But
      should be added for sure.
"""
from typing import List

import pytest
from numpy import testing
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

    testing.assert_equal(output_data[attrs].to_numpy().round(4), data_input_restored[attrs].to_numpy().round(4))


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
