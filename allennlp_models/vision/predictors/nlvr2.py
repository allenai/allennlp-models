from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.predictors.predictor import Predictor


@Predictor.register("nlvr2")
class Nlvr2Predictor(Predictor):
    def predict(self, image1: str, image2: str, hypothesis: str) -> JsonDict:
        image1 = cached_path(image1)
        image2 = cached_path(image2)
        return self.predict_json({"image1": image1, "image2": image2, "hypothesis": hypothesis})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        from allennlp_models.vision.dataset_readers.nlvr2 import Nlvr2Reader

        image1 = cached_path(json_dict["image1"])
        image2 = cached_path(json_dict["image2"])
        hypothesis = json_dict["hypothesis"]
        if isinstance(self._dataset_reader, Nlvr2Reader):
            return self._dataset_reader.text_to_instance(
                hypothesis, image1, image2, use_cache=False
            )
        else:
            raise ValueError(
                f"Dataset reader is of type f{self._dataset_reader.__class__.__name__}. "
                f"Expected {Nlvr2Reader.__name__}."
            )

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
