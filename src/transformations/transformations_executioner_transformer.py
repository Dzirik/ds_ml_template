"""
Transformer
"""

from typing import Any, List, Dict, NamedTuple, Union, Optional, Type, cast

from numpy import float64
from pandas import DataFrame

from src.exceptions.data_exception import MismatchedDimension, IncorrectDataStructure
from src.exceptions.development_exception import NotValidOperation
from src.transformations.base_transformer import BaseTransformer, TransformerDescription
from src.transformations.difference_percentage_change_transformer import DifferencePercentageChangeTransformer
from src.transformations.difference_transformer import DifferenceTransformer
from src.transformations.log_negative_transformer import LogNegativeTransformer
from src.transformations.log_transformer import LogTransformer
from src.transformations.min_max_scaling_transformer import MinMaxScalingTransformer
from src.transformations.quantile_transformer import QuantileTransformer
from src.transformations.shift_by_value_transformer import ShiftByValueTransformer

# Type alias for transformer classes
TransformerClassType = Type[Union[DifferenceTransformer, DifferencePercentageChangeTransformer,
                                   LogTransformer, LogNegativeTransformer, MinMaxScalingTransformer,
                                   QuantileTransformer, ShiftByValueTransformer]]


class TransformerConfiguration(NamedTuple):
    """
    For storing parameters for one transformer for transformation executioner.
    - name - str. Name of the transformer to be created.
    - fit - bool. If fit or not. If not, params are taken.
    - params_for_fit - Dict[str, Any]. Parameters for fit.
    - params_fitted - Dict[str, Any]. Fitted parameters.
    """
    name: str
    fit: bool
    params_for_fit: Dict[str, Any]
    params_fitted: Dict[str, Any]


class TransformationsExecutionerTransformer(BaseTransformer):  # type: ignore[misc]
    """
    A complex transformer for executing a list of transformations on selected column(s) together with their inverse.
    If all the transformers are invertible, inversion is possible.
    Checks the dimensionality of transformers and data (some ts are only for one column).
    NOTE: The logic of transformations is kept on the user's sense!
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(
            input_type=[Any], input_elements_type=[Any],
            output_type=[Any], output_elements_type=[Any]
        )
        BaseTransformer.__init__(
            self, class_name="TransformationsExecutionerTransformer", transformer_description=transformer_description
        )

        self._attrs: List[str] = []
        self._configurations: List[TransformerConfiguration] = []
        self._is_invertible: bool = True
        self._transformers: List[Union[DifferenceTransformer, DifferencePercentageChangeTransformer,
                                       LogTransformer, LogNegativeTransformer, MinMaxScalingTransformer,
                                       QuantileTransformer, ShiftByValueTransformer]] = []

        # "": {"class": , "invertible": },
        self._transformers_dictionary: Dict[str, Dict[str, Union[TransformerClassType, bool]]] = {
            "DifferenceTransformer": {"class": DifferenceTransformer, "invertible": False, "one_dim": True},
            "DifferencePercentageChangeTransformer": {
                "class": DifferencePercentageChangeTransformer, "invertible": False, "one_dim": True
            },
            "LogTransformer": {"class": LogTransformer, "invertible": True, "one_dim": False},
            "LogNegativeTransformer": {"class": LogNegativeTransformer, "invertible": True, "one_dim": False},
            "MinMaxScalingTransformer": {"class": MinMaxScalingTransformer, "invertible": True, "one_dim": False},
            "QuantileTransformer": {"class": QuantileTransformer, "invertible": False, "one_dim": True},
            "ShiftByValueTransformer": {"class": ShiftByValueTransformer, "invertible": True, "one_dim": False},
        }

    # pylint: disable=arguments-differ
    def fit(self, data: Optional[DataFrame], attrs: List[str], configurations: List[TransformerConfiguration]) -> None:
        """
        Fits.
        - Creates a list of transformers from the configuration and for each transformer:
            - Fit it or restore from fitted parameters.
            - If fit is done, the fitted parameters are stored into the configuration.
        If data is None, then the configuration has to have set up for "fit" = False choice and just restore the
        transformers.
        :param data: Optional[DataFrame]. Data frame to be used for fitting.
        :param attrs: List[str]. List of attributes of the data frame to be transformed.
        :param configurations: List[TransformerConfiguration]. Configuration of the set of transformations to be
                               applied.
        """
        self._attrs = attrs
        self._configurations = configurations
        self._is_invertible = True

        self._transformers = []
        # Maintain a working copy of attribute data so each transformer fits on the
        # output of the previous one, mirroring the sequential logic in predict().
        working_data = data[self._attrs].to_numpy(dtype=float64) if data is not None else None
        for i, conf in enumerate(self._configurations):
            if self._transformers_dictionary[conf.name]["one_dim"] and len(attrs) > 1:
                raise MismatchedDimension
            transformer_class = cast(TransformerClassType,
                                     self._transformers_dictionary[conf.name]["class"])
            t = transformer_class()
            invertible = cast(bool, self._transformers_dictionary[conf.name]["invertible"])
            self._is_invertible = self._is_invertible and invertible
            if conf.fit:
                if working_data is not None:
                    t.fit(working_data, **conf.params_for_fit)
                    self._configurations[i] = conf._replace(fit=False, params_fitted=t.get_params())
                    working_data = t.predict(working_data)
                elif conf.params_fitted:
                    t.restore_from_params(conf.params_fitted)
                    self._configurations[i] = conf._replace(fit=False)
                else:
                    raise IncorrectDataStructure
            else:
                t.restore_from_params(conf.params_fitted)
                if working_data is not None:
                    working_data = t.predict(working_data)
            self._transformers.append(t)

    def fit_predict(self, data: Optional[DataFrame], attrs: List[str], configurations: List[TransformerConfiguration]) \
            -> DataFrame:
        """
        Fits and predicts.
        :param data: Optional[DataFrame]. Data frame to be used for fitting.
        :param attrs: List[str]. List of attributes of the data frame to be transformed.
        :param configurations: List[TransformerConfiguration]. Configuration of the set of transformations to be
                               applied.
        :return: DataFrame.
        """
        self.fit(data, attrs, configurations)
        return self.predict(data)

    def predict(self, data: DataFrame) -> DataFrame:
        """
        Predicts.
        :param data: DataFrame.
        :return: DataFrame.
        """
        data = data.copy()
        for t in self._transformers:
            data[self._attrs] = t.predict(data[self._attrs].to_numpy(dtype=float64))
        return data

    def inverse(self, data: DataFrame) -> Any:
        """
        Does the inverse transformation if possible. If not, raises and exception.
        :param data: DataFrame.
        :return: DataFrame.
        """
        if self._is_invertible:
            data = data.copy()
            for t in reversed(self._transformers):
                # Only invertible transformers (LogTransformer, LogNegativeTransformer, MinMaxScalingTransformer,
                # ShiftByValueTransformer) reach here, and they all accept a data argument.
                data[self._attrs] = t.inverse(data[self._attrs].to_numpy(dtype=float64))
            return data
        raise NotValidOperation

    # pylint: enable=arguments-differ

    def get_params(self) -> Dict[str, Any]:
        """
        Gets the params.
        :return: Dict[str, Any]. Params.
        """
        conf_dicts = []
        for conf in self._configurations:
            conf_dicts.append(conf._asdict())
        return {
            "attrs": self._attrs,
            "configurations": conf_dicts,
        }

    def restore_from_params(self, params: Dict[str, Any]) -> None:
        """
        Sets the params for transformer.
        :param params: Dict[str, Any]
        """
        self._attrs = params["attrs"]

        self._configurations = []
        self._transformers = []
        self._is_invertible = True
        for d in params["configurations"]:
            conf = TransformerConfiguration(**d)
            transformer_class = cast(TransformerClassType,
                                     self._transformers_dictionary[conf.name]["class"])
            t = transformer_class()
            invertible = cast(bool, self._transformers_dictionary[conf.name]["invertible"])
            self._is_invertible = self._is_invertible and invertible
            t.restore_from_params(conf.params_fitted)

            self._configurations.append(conf)
            self._transformers.append(t)
