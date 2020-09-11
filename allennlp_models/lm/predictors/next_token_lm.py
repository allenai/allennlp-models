from typing import Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.predictors.predictor import Predictor


@Predictor.register("next_token_lm")
class NextTokenLMPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ):
        new_instance = instance.duplicate()
        token_field: TextField = instance["tokens"]  # type: ignore
        mask_targets = [
            Token(target_top_k_text[0], text_id=target_top_id_id)
            for (target_top_k_text, target_top_id_id) in zip(outputs["words"], outputs["token_ids"])
        ]

        new_instance.add_field(
            "target_ids",
            TextField(mask_targets, token_field._token_indexers),
            vocab=self._model.vocab,
        )
        return [new_instance]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        """
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence=sentence)  # type: ignore
