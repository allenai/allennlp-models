import os
import glob
from typing import Dict
from allennlp.common import Params
from allennlp_models.common.model_card import ModelCard


def get_pretrained_models() -> Dict[str, ModelCard]:
    """
    Returns a Dict of model cards of all available pretrained models.
    """

    # These imports are included so that the model cards can be filled with default information
    # obtained from the registered model classes.

    from allennlp_models.classification import models as classification_models  # noqa: F401
    from allennlp_models.coref import models as coref_models  # noqa: F401
    from allennlp_models.generation import models as generation_models  # noqa: F401
    from allennlp_models.lm import models as lm_models  # noqa: F401
    from allennlp_models.mc import models as mc_models  # noqa: F401
    from allennlp_models.pair_classification import models as pc_models  # noqa: F401
    from allennlp_models.rc import models as rc_models  # noqa: F401
    from allennlp_models.structured_prediction import models as sp_models  # noqa: F401
    from allennlp_models.tagging import models as tagging_models  # noqa: F401

    pretrained_models = {}
    modelcards = os.path.join(os.path.dirname(os.path.realpath(__file__)), "modelcards/*.json")

    for json_modelcard in glob.glob(modelcards):
        model_card = ModelCard.from_params(params=Params.from_file(json_modelcard))
        pretrained_models[model_card.name] = model_card
    return pretrained_models
