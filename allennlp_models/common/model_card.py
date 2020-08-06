import os
import logging
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
from allennlp.common.from_params import FromParams

from allennlp.models import Model
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


def get_description(model_class):
    """
    Returns the model's description from the docstring.
    """
    return model_class.__doc__.split("# Parameters")[0].strip()


class ModelCardInfo(FromParams):
    def to_dict(self):
        """
        Only the non-empty attributes are returned, to minimize empty values.
        """
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info[key] = val
        return info

    def __str__(self):
        display = ""
        for key, val in self.to_dict().items():
            display += "\n" + key.replace("_", " ").capitalize() + ": "
            display += "\n\t" + val.replace("\n", "\n\t") + "\n"
        if not display:
            display = super(ModelCardInfo, self).__str__()
        return display.strip()


@dataclass(frozen=True)
class ModelDetails(ModelCardInfo):
    """
    This provides the basic information about the model.
    """

    description: Optional[str] = None
    developed_by: Optional[str] = None
    contributed_by: Optional[str] = None
    date: Optional[str] = None
    version: Optional[str] = None
    model_type: Optional[str] = None
    paper: Optional[str] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    contact: Optional[str] = None


@dataclass(frozen=True)
class IntendedUse(ModelCardInfo):
    """
    This determines what the model should and should not be
    used for.
    """

    primary_uses: Optional[str] = None
    primary_users: Optional[str] = None
    out_of_scope_use_cases: Optional[str] = None


@dataclass(frozen=True)
class Factors(ModelCardInfo):
    """
    This provides a summary of relevant factors such as
    demographics, instrumentation used, etc. for which the
    model performance may vary.
    """

    relevant_factors: Optional[str] = None
    evaluation_factors: Optional[str] = None


@dataclass(frozen=True)
class Metrics(ModelCardInfo):
    """
    This lists the reported metrics and the reasons
    for choosing them.
    """

    model_performance_measures: Optional[str] = None
    decision_thresholds: Optional[str] = None
    variation_approaches: Optional[str] = None


@dataclass(frozen=True)
class EvaluationData(ModelCardInfo):
    """
    This provides information about the evaluation data.
    """

    dataset: Optional[str] = None
    motivation: Optional[str] = None
    preprocessing: Optional[str] = None

    def to_dict(self):
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info["evaluation_" + key] = val
        return info


@dataclass(frozen=True)
class TrainingData(ModelCardInfo):
    """
    This provides information about the training data.
    """

    dataset: Optional[str] = None
    motivation: Optional[str] = None
    preprocessing: Optional[str] = None

    def to_dict(self):
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info["training_" + key] = val
        return info


@dataclass(frozen=True)
class QuantitativeAnalyses(ModelCardInfo):
    """
    This provides disaggregated evaluation of how the
    model performed based on chosen metrics.
    """

    unitary_results: Optional[str] = None
    intersectional_results: Optional[str] = None


@dataclass(frozen=True)
class EthicalConsiderations(ModelCardInfo):
    """
    This highlights any ethical considerations to keep
    in mind when using the model.
    Eg. Is the model intended to be used for informing
    decisions on human life? Does it use sensitive data?
    What kind of risks are possible, and what mitigation
    strategies were used to address them?
    """

    ethical_considerations: Optional[str] = None


@dataclass(frozen=True)
class CaveatsAndRecommendations(ModelCardInfo):
    """
    This lists any additional concerns. Eg. were any relevant
    groups not present in the evaluation data?
    """

    caveats_and_recommendations: Optional[str] = None


class ModelCard(ModelCardInfo):
    """
    The model card stores the recommended attributes for model reporting
    as described in the paper
    [Model Cards for Model Reporting (Mitchell et al, 2019)](https://arxiv.org/pdf/1810.03993.pdf).

    # Parameters

    id: `str`
        Model's id, following the convention of task-model-relevant-details.
        Example: rc-bidaf-elmo for a reading comprehension BiDAF model using ELMo embeddings.
    registered_model_name: `str`, optional
        The model's registered name. If `model_class` is not given, this will be used
        to find any available `Model` registered with this name.
    model_class: `type`, optional
        If given, the `ModelCard` will pull some default information from the class.
    registered_predictor_name: `str`, optional
        The registered name of the corresponding predictor.
    display_name: `str`, optional
        The pretrained model's display name.
    archive_file: `str`, optional
        The location of model's pretrained weights.
    overrides: `Dict`, optional
        Optional overrides for the model's architecture.
    model_details: `Union[ModelDetails, str]`, optional
    intended_use: `Union[IntendedUse, str]`, optional
    factors: `Union[Factors, str]`, optional
    metrics: `Union[Metrics, str]`, optional
    evaluation_data: `Union[EvaluationData, str]`, optional
    quantitative_analyses: `Union[QuantitativeAnalyses, str]`, optional
    ethical_considerations: `Union[EthicalConsiderations, str]`, optional
    caveats_and_recommendations: `Union[CaveatsAndRecommendations, str]`, optional

    !!! Note
        For all the fields that are `Union[ModelCardInfo, str]`, a `str` input will be
        treated as the first argument of the relevant constructor.

    """

    _storage_location = "https://storage.googleapis.com/allennlp-public-models/"

    def __init__(
        self,
        id: str,
        registered_model_name: Optional[str] = None,
        model_class: Optional[type] = None,
        registered_predictor_name: Optional[str] = None,
        display_name: Optional[str] = None,
        archive_file: Optional[str] = None,
        overrides: Optional[Dict] = None,
        model_details: Optional[Union[str, ModelDetails]] = None,
        intended_use: Optional[Union[str, IntendedUse]] = None,
        factors: Optional[Union[str, Factors]] = None,
        metrics: Optional[Union[str, Metrics]] = None,
        evaluation_data: Optional[Union[str, EvaluationData]] = None,
        training_data: Optional[Union[str, TrainingData]] = None,
        quantitative_analyses: Optional[Union[str, QuantitativeAnalyses]] = None,
        ethical_considerations: Optional[Union[str, EthicalConsiderations]] = None,
        caveats_and_recommendations: Optional[Union[str, CaveatsAndRecommendations]] = None,
    ):

        assert id
        if not model_class and registered_model_name:
            try:
                model_class = Model.by_name(registered_model_name)
            except ConfigurationError:
                logger.warning("{} is not a registered model.".format(registered_model_name))

        if model_class:
            display_name = display_name or model_class.__name__
            model_details = model_details or get_description(model_class)
            if not registered_predictor_name:
                registered_predictor_name = model_class.default_predictor  # type: ignore

        if archive_file and not archive_file.startswith("https:"):
            archive_file = os.path.join(self._storage_location, archive_file)

        if isinstance(model_details, str):
            model_details = ModelDetails(description=model_details)
        if isinstance(intended_use, str):
            intended_use = IntendedUse(primary_uses=intended_use)
        if isinstance(factors, str):
            factors = Factors(relevant_factors=factors)
        if isinstance(metrics, str):
            metrics = Metrics(model_performance_measures=metrics)
        if isinstance(evaluation_data, str):
            evaluation_data = EvaluationData(dataset=evaluation_data)
        if isinstance(training_data, str):
            training_data = TrainingData(dataset=training_data)
        if isinstance(quantitative_analyses, str):
            quantitative_analyses = QuantitativeAnalyses(unitary_results=quantitative_analyses)
        if isinstance(ethical_considerations, str):
            ethical_considerations = EthicalConsiderations(ethical_considerations)
        if isinstance(caveats_and_recommendations, str):
            caveats_and_recommendations = CaveatsAndRecommendations(caveats_and_recommendations)

        self.id = id
        self.registered_model_name = registered_model_name
        self.registered_predictor_name = registered_predictor_name
        self.display_name = display_name
        self.archive_file = archive_file
        self.model_details = model_details
        self.intended_use = intended_use
        self.factors = factors
        self.metrics = metrics
        self.evaluation_data = evaluation_data
        self.training_data = training_data
        self.quantitative_analyses = quantitative_analyses
        self.ethical_considerations = ethical_considerations
        self.caveats_and_recommendations = caveats_and_recommendations

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the `ModelCard` to a flat dictionary object. This can be converted to
        json and passed to any front-end.
        """
        info = {}
        for key, val in self.__dict__.items():
            if key != "id":
                if isinstance(val, ModelCardInfo):
                    info.update(val.to_dict())
                else:
                    if val is not None:
                        info[key] = val
        return info
