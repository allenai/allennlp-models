from typing import List
import json
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register("squad_question_generation")
class QuestionGenerationPredictor(Predictor):
    def predict(
        self,
        passage: str,
        answer_text: str,
        answer_start_charoffsets: List[int],
        masked_question: str = None,
        passage_id: str = None,
        query_id: str = None,
    ):
        json_dict = {
            "passage": passage,
            "answer_text": answer_text,
            "answer_start_charoffsets": answer_start_charoffsets,
        }

        if masked_question is not None:
            json_dict.update({"masked_question": masked_question})

        if passage_id is not None:
            json_dict.update({"passage_id": passage_id})

        if query_id is not None:
            json_dict.update({"query_id": query_id})

        # Calls json_to_instance -> predict_instance -> Returns `outputs` from model
        return self.predict_json(inputs=json_dict)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        passage: str = json_dict["passage"]
        answer_text: str = json_dict["answer_text"]
        answer_start_charoffsets: List[int] = json_dict["answer_start_charoffsets"]
        passage_id = json_dict.get("passage_id", None)
        query_id = json_dict.get("query_id", None)

        return self._dataset_reader.text_to_instance(
            passage, answer_text, answer_start_charoffsets, passage_id, query_id
        )

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        metadata = outputs["metadata"]

        output_dict = {
            "predicted_question": outputs["predicted_question"],
            "answer": metadata["answer"],
            "passage": metadata["passage"],
            "ans_marked_passage": metadata["ans_marked_passage"],
            "answer_start": metadata["answer_start"],
            "answer_end": metadata["answer_end"],
        }

        return json.dumps(output_dict) + "\n"
