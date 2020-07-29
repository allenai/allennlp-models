import os
import glob
from typing import Dict
from allennlp.common import Params
from allennlp_models.common.model_card import ModelCard

# These imports are included so that the model cards can be filled with default information
# obtained from the registered model classes.

from allennlp_models.classification.models import *  # noqa: F401, F403
from allennlp_models.coref.models import *  # noqa: F401, F403
from allennlp_models.generation.models import *  # noqa: F401, F403
from allennlp_models.lm.models import *  # noqa: F401, F403
from allennlp_models.mc.models import *  # noqa: F401, F403
from allennlp_models.pair_classification.models import *  # noqa: F401, F403
from allennlp_models.rc.models import *  # noqa: F401, F403
from allennlp_models.structured_prediction.models import *  # noqa: F401, F403
from allennlp_models.tagging.models import *  # noqa: F401, F403


def get_pretrained_models() -> Dict[str, ModelCard]:
    """
    Returns a Dict of model cards of all available pretrained models.
    """

    pretrained_models = {}
    model_card_paths = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "modelcards", "*.json"
    )

    for model_card_path in glob.glob(model_card_paths):
        model_card = ModelCard.from_params(params=Params.from_file(model_card_path))
        pretrained_models[model_card.id] = model_card
    return pretrained_models
