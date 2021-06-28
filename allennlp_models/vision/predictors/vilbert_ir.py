from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.predictors.predictor import Predictor


@Predictor.register("vilbert_ir")
class VilbertImageRetrievalPredictor(Predictor):
    def predict(self, image: str, caption: str) -> JsonDict:
        image = cached_path(image)
        return self.predict_json({"caption": caption, "image": image})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        from allennlp_models.vision.dataset_readers.flickr30k import Flickr30kReader

        caption = json_dict["caption"]
        image = cached_path(json_dict["image"])
        if isinstance(self._dataset_reader, Flickr30kReader):
            return self._dataset_reader.text_to_instance(caption, image, use_cache=False)
        else:
            raise ValueError(
                f"Dataset reader is of type f{self._dataset_reader.__class__.__name__}. "
                f"Expected {Flickr30kReader.__name__}."
            )

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
