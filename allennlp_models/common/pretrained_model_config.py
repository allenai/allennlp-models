import os
import weakref
from dataclasses import dataclass
from typing import Optional, Mapping, Dict
from collections import defaultdict

from allennlp.models import Model
from allennlp.common.checks import ConfigurationError


def get_description(model_class):
    """
    Returns the model's description from the docstring.
    """
    return model_class.__doc__.split("# Parameters")[0].strip()


@dataclass(frozen=True)
class ModelCard(object):
    pass


class PretrainedModelConfiguration(object):

    __refs__ = defaultdict(list)  # type: ignore
    _storage_location = "https://storage.googleapis.com/allennlp-public-models/"

    def __init__(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        last_trained_on: Optional[str] = None,
        archive_file: Optional[str] = None,
        predictor_name: Optional[str] = None,
        overrides: Optional[Mapping] = None,
        model_card: Optional[ModelCard] = None,
    ):

        self.id = id
        self.archive_file = archive_file
        if self.archive_file and not self.archive_file.startswith("https:"):
            self.archive_file = os.path.join(self._storage_location, self.archive_file)
        self.name = name
        self.description = description
        self.last_trained_on = last_trained_on
        self.predictor_name = predictor_name
        self.overrides = overrides
        self.model_card = model_card

        self.__refs__[self.__class__].append(weakref.ref(self))

    @staticmethod
    def from_dict(config: Dict, model_class: Optional[Model] = None):
        assert "id" in config
        if not model_class:
            try:
                model_class = Model.by_name(config["id"])
            except ConfigurationError:
                # TODO: log stuff.
                pass
        if model_class:
            config["name"] = config.get("name", model_class.__name__)
            config["description"] = config.get("description", get_description(model_class))
            config["predictor_name"] = config.get("predictor_name", model_class.default_predictor)

        # config["last_trained_on"] = config.get("last_trained_on", "str stuff here")
        return PretrainedModelConfiguration(**config)

    @classmethod
    def get_instances(cls):
        for inst_ref in cls.__refs__[cls]:
            inst = inst_ref()
            if inst is not None:
                yield inst
