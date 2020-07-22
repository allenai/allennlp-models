import os
import logging
from dataclasses import dataclass
from typing import Optional, Union, Mapping

from allennlp.models import Model
from allennlp.common.checks import ConfigurationError

PRETRAINED_MODELS = {}
STORAGE_LOCATION = "https://storage.googleapis.com/allennlp-public-models/"

logger = logging.getLogger(__name__)

def get_description(model_class):
    """
    Returns the model's description from the docstring.
    """
    return model_class.__doc__.split("# Parameters")[0].strip()

@dataclass
class ModelCardInfo(object):
    """
    Base class for different recommended attributes included
    in a model card.
    """
    @classmethod
    def from_object(cls, obj: Union[str, "ModelCardInfo"]):
        """
        Creates the relevant `ModelCardInfo` obj. If the input is str,
        it is initialized as the first attribute of the class.
        """
        if isinstance(obj, str):
            return cls(obj)
        return obj
    
    def to_dict(self):
        """
        Only the non-empty attributes are returned, to minimize empty values.
        """
        info = {}
        for key, val in self.__dict__.items():
            if val:
                info[key] = val
        return info

@dataclass
class ModelDetails(ModelCardInfo):
    """
    This provides the basic information about the model.
    """
    description: Optional[str] = None
    developed_by: Optional[str] = None
    date: Optional[str] = None
    version: Optional[str] = None
    model_type: Optional[str] = None
    paper: Optional[str] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    contact: Optional[str] = None

@dataclass
class IntendedUse(ModelCardInfo):
    """
    This determines what the model should and should not be
    used for.
    """
    primary_uses: Optional[str] = None
    primary_users: Optional[str] = None
    out_of_scope_uses_cases: Optional[str] = None

@dataclass
class Factors(ModelCardInfo):
    """
    This provides a summary of relevant factors such as
    demographics, instrumentation used, etc. for which the
    model performance may vary.
    """
    relevant_factors: Optional[str] = None
    evaluation_factors: Optional[str] = None

@dataclass
class Metrics(ModelCardInfo):
    """
    This lists the reported metrics and the reasons
    for choosing them.
    """
    model_performance_measures: Optional[str] = None
    decision_thresholds: Optional[str] = None
    variation_approaches: Optional[str] = None
        
@dataclass
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
        
@dataclass
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
        
@dataclass
class QuantitativeAnalyses(ModelCardInfo):
    """
    This provides disaggregated evaluation of how the
    model performed based on chosen metrics.
    """
    unitary_results: Optional[str] = None
    intersectional_results: Optional[str] = None
        
@dataclass
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
        
@dataclass
class CaveatsAndRecommendations(ModelCardInfo):
    """
    This lists any additional concerns. Eg. were any relevant
    groups not present in the evaluation data?
    """
    caveats_and_recommendations: Optional[str] = None


@dataclass(frozen=True)
class ModelCard(ModelCardInfo):
    """
    The model card stores the recommended attributes for model reporting
    as described in the paper [Model Cards for Model Reporting (Mitchell et al.)] 
    (https://arxiv.org/pdf/1810.03993.pdf).
    """
    id: str
    name: Optional[str] = None
    archive_file: Optional[str] = None
    predictor_name: Optional[str] = None
    overrides: Optional[Mapping] = None
    model_details: Optional[ModelDetails] = None
    intended_use: Optional[IntendedUse] = None
    factors: Optional[Factors] = None
    metrics: Optional[Metrics] = None
    evaluation_data: Optional[EvaluationData] = None
    training_data: Optional[TrainingData] = None
    quantitative_analyses: Optional[QuantitativeAnalyses] = None
    ethical_considerations: Optional[EthicalConsiderations] = None
    caveats_and_recommendations: Optional[CaveatsAndRecommendations] = None
        
    def to_dict(self):
        info = {}
        for key, val in self.__dict__.items():
            if isinstance(val, ModelCardInfo):
                info.update(val.to_dict())
            else:
                info[key] = val
        return info

def add_pretrained_model(id: str,
                         model_class: Optional[type] = None,
                         name: Optional[str] = None,
                         archive_file: Optional[str] = None,
                         predictor_name: Optional[str] = None,
                         overrides: Optional[Mapping] = None,
                         model_details: Optional[Union[str, ModelDetails]] = None,
                         intended_use: Optional[Union[str, IntendedUse]] = None,
                         factors: Optional[Union[str, Factors]] = None,
                         metrics: Optional[Union[str, Metrics]] = None,
                         evaluation_data: Optional[Union[str, EvaluationData]] = None,
                         training_data: Optional[Union[str, TrainingData]] = None,
                         quantitative_analyses: Optional[Union[str, QuantitativeAnalyses]] = None,
                         ethical_considerations: Optional[Union[str, EthicalConsiderations]] = None,
                         caveats_and_recommendations: Optional[Union[str, CaveatsAndRecommendations]] = None
                        ) -> ModelCard:
    """
    Creates a `ModelCard` object and registers it
    in the global dict of available pretrained models.
    """
    assert id
    config = {}
    config["id"] = id
    if not model_class:
        try:
            model_class = Model.by_name(id)
        except ConfigurationError:
            logger.warning("{} is not a registered model.".format(id))
    
    if model_class:
        name = name or model_class.__name__
        predictor_name = predictor_name or model_class.default_predictor
        model_details = model_details or get_description(model_class)

    if archive_file and not archive_file.startswith("https:"):
        archive_file = os.path.join(STORAGE_LOCATION, archive_file)

    config["name"] = name
    config["predictor_name"] = predictor_name
    config["archive_file"] = archive_file
        
    config["model_details"] = ModelDetails.from_object(model_details)
    config["intended_use"] = IntendedUse.from_object(intended_use)
    config["factors"] = Factors.from_object(factors)
    config["metrics"] = Metrics.from_object(metrics)
    config["evaluation_data"] = EvaluationData.from_object(evaluation_data)
    config["training_data"] = TrainingData.from_object(training_data)
    config["quantitative_analyses"] = QuantitativeAnalyses.from_object(quantitative_analyses)
    config["ethical_considerations"] = EthicalConsiderations.from_object(ethical_considerations)
    config["caveats_and_recommendations"] = CaveatsAndRecommendations.from_object(caveats_and_recommendations)

    model_card = ModelCard(**config)
    PRETRAINED_MODELS[model_card.id] = model_card
    return model_card
