from typing import Dict

from allennlp.predictors import Predictor

from allennlp_models.classification import models as classification_models  # noqa: F401
from allennlp_models.coref import models as coref_models  # noqa: F401
from allennlp_models.generation import models as generation_models  # noqa: F401
from allennlp_models.lm import models as lm_models  # noqa: F401
from allennlp_models.mc import models as mc_models  # noqa: F401
from allennlp_models.pair_classification import models as pc_models  # noqa: F401
from allennlp_models.rc import models as rc_models  # noqa: F401
from allennlp_models.structured_prediction import models as sp_models  # noqa: F401
from allennlp_models.tagging import models as tagging_models  # noqa: F401

from allennlp_models.common.model_card import get_pretrained_models

# TODO: rename all archive files to follow standard naming.
# Standard: task-model.yyyy-mm-dd.tar.gz OR model-task.yyyy-mm-dd.tar.gz

PRETRAINED_MODELS = get_pretrained_models()

PRETRAINED_MODELS["glove-sst"] = {
    "archive_file": "basic_stanford_sentiment_treebank-2020.06.09.tar.gz"
}

PRETRAINED_MODELS["roberta-sst"] = {"archive_file": "sst-roberta-large-2020.06.08.tar.gz"}


def get_model_config(model: str) -> Dict:
    try:
        return PRETRAINED_MODELS[model]
    except KeyError:
        # TODO: better error message.
        raise RuntimeError("The model {} is not available".format(model))


def get_predictor(model: str) -> Predictor:
    """
    Returns the model's predictor object.
    """
    try:
        archive_file = PRETRAINED_MODELS[model]["archive_file"]
        predictor_name = PRETRAINED_MODELS[model].get("predictor_name")
        return Predictor.from_path(archive_file, predictor_name)
    except KeyError:
        raise RuntimeError("An archive is not available for {}".format(model))
