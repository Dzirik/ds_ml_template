"""
Model based on transformer.
"""
from typing import Any, NamedTuple, Dict, List, Union

from numpy import ndarray, dtype, double, std, apply_along_axis

from src.exceptions.data_exception import IncorrectDataStructure
from src.models.anomaly_detection_rules import FirstWecoRule, SecondWecoRule, ThirdWecoRule, FourthWecoRule
from src.transformations.base_transformer import BaseTransformer, TransformerDescription


class DetectionRuleParams(NamedTuple):
    """
    Named tuple for storing data for one rule.
    - name: str. Name of the rule.
    - window_len: int. Exact number of residuals needed for the rule.
    - params: Dict[str, Any]. Dictionary of parameters.
    """
    name: str
    params: Dict[str, Any]


class AnomalyDetectionParams(NamedTuple):
    """
    Named tuple for storing data for the whole model.
    """
    rules_definition: List[DetectionRuleParams]


class AnomalyDetectionRulesModel(BaseTransformer):  # type:ignore
    """
    Anomaly detection rule based model aka WECO rules and extension.
    """

    def __init__(self) -> None:
        transformer_description = TransformerDescription(
            input_type=[ndarray[Any, dtype[double]]], input_elements_type=[ndarray[Any, dtype[double]]],
            output_type=[ndarray[Any, dtype[double]]], output_elements_type=[ndarray[Any, dtype[double]]]
        )
        BaseTransformer.__init__(
            self, class_name="AnomalyDetectionRulesModel", transformer_description=transformer_description
        )

        self._rules_dictionary = {
            "FirstWecoRule": {"class": FirstWecoRule, "add_std_est": True},
            "SecondWecoRule": {"class": SecondWecoRule, "add_std_est": True},
            "ThirdWecoRule": {"class": ThirdWecoRule, "add_std_est": True},
            "FourthWecoRule": {"class": FourthWecoRule, "add_std_est": True}
        }

        # external settings
        self._anomaly_detection_params: AnomalyDetectionParams
        self._std_est: float

        # constructed from _params
        self._rule_names: List[str]
        self._rule_param_dicts: List[Dict[str, Any]]
        self._rules: List[Union[FirstWecoRule, SecondWecoRule, ThirdWecoRule, FourthWecoRule]]

    def get_rule_names(self) -> List[str]:
        """
        Gets the rule names.
        :return: List[str].
        """
        return self._rule_names

    def _construct_rules(self) -> None:
        """
        Constructs the rules
        """
        self._rule_names = []
        self._rules = []
        self._rule_param_dicts = []
        for rule_definition in self._anomaly_detection_params.rules_definition:
            self._rule_names.append(rule_definition.name)

            self._rules.append(self._rules_dictionary[rule_definition.name]["class"]())

            rule_param = rule_definition.params
            rule_param["std_est"] = self._std_est
            self._rule_param_dicts.append(rule_param)

    def _apply_rules_for_residuals(self, residuals: ndarray[Any, dtype[double]]) -> List[int]:
        """
        :param residuals: ndarray[Any, dtype[double]]. One dimensional numpy array.
        :return: List[int]. List of 0 - no anomaly, 1 - is anomaly for rule in self._rule_names.
        """
        results = []
        for rule, param_dictionary in zip(self._rules, self._rule_param_dicts):
            n = rule.get_window_len()
            results.append(rule.apply(
                residuals[-n:],
                **param_dictionary
            ))
        return results

    # pylint: disable=arguments-differ
    def fit(self, data_fit: ndarray[Any, dtype[double]], params: AnomalyDetectionParams) -> None:
        """
        Fits.

        Tests shape.
        :param data_fit: ndarray[Any, dtype[double]]. Two dimensional array of the shape (n, 1).
        :param params: AnomalyDetectionParams. Parameters for the model.
        """
        if data_fit.shape[1] != 1:
            raise IncorrectDataStructure

        self._anomaly_detection_params = params

        self._std_est = std(data_fit)

        self._construct_rules()

    def fit_predict(self, data_fit: ndarray[Any, dtype[double]], data_predict: ndarray[Any, dtype[double]], \
                    params: AnomalyDetectionParams) -> ndarray[Any, dtype[double]]:
        """
        Fits and predicts.
        :param data_fit: ndarray[Any, dtype[double]]. Two dimensional array of the shape (n, 1).
        :param data_predict: ndarray[Any, dtype[double]]. Two dimensional array of the shape (m, p). Rows are
            sequences of residuals to be tested.
        :param params: AnomalyDetectionParams. Parameters for the model.
        :return: ndarray[Any, dtype[double]].
        """
        self.fit(data_fit, params)
        return self.predict(data_predict)

    # pylint: disable=arguments-renamed
    def predict(self, data_predict: ndarray[Any, dtype[double]]) -> ndarray[Any, dtype[double]]:
        """
        Predicts.
        :param data_predict: ndarray[Any, dtype[double]]. Two dimensional array of the shape (m, p). Rows are
            sequences of residuals to be tested.
        :return: ndarray[Any, dtype[double]]. Contains 0 - no anomaly for the rule, 1 - anomaly for the rule.
        """
        return apply_along_axis(self._apply_rules_for_residuals, 1, data_predict)

    # pylint: enable=arguments-renamed

    def inverse(self) -> None:
        """
        Does the inverse transformation.
        """
        return None

    # pylint: enable=arguments-differ

    def get_params(self) -> Dict[str, Any]:
        """
        Gets the params.
        :return: Dict[str, Any].
        """
        rules_definition = [named_tuple._asdict() for named_tuple in self._anomaly_detection_params.rules_definition]
        return {
            "std_est": self._std_est,
            "rules_definition": rules_definition
        }

    def restore_from_params(self, params: Dict[str, Any]) -> None:
        """
        Sets the params for transformer.
        :param params: Dict[str, Any].
        """
        self._std_est = params["std_est"]
        self._anomaly_detection_params = AnomalyDetectionParams(
            rules_definition=[DetectionRuleParams(**d) for d in params["rules_definition"]]
        )
        self._construct_rules()
