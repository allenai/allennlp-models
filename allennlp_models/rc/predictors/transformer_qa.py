import logging
from typing import List, Dict, Any, Optional

from overrides import overrides
import numpy

from allennlp.models import Model
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.predictors.predictor import Predictor

from allennlp.data.fields import SpanField

logger = logging.getLogger(__name__)


@Predictor.register("transformer_qa")
class TransformerQAPredictor(Predictor):
    """
    Predictor for the [`TransformerQA`](/models/rc/models/transformer_qa#transformer_qa) model,
    and any other model that takes a question and passage as input.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(TransformerQAPredictor, self).__init__(model, dataset_reader)
        self._next_qid = 1

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
        for more information about the machine comprehension task.

        # Parameters

        question : `str`
            A question about the content in the supplied paragraph.

        passage : `str`
            A paragraph of information relevant to the question.

        # Returns

        `JsonDict`
            A dictionary that represents the prediction made by the system.
            The answer string will be under the `"best_span_str"` key.

        """
        return self.predict_json({"context": passage, "question": question})

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        results = self.predict_batch_json([inputs])
        assert len(results) == 1
        return results[0]

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        span_start = int(outputs["best_span"][0])
        span_end = int(outputs["best_span"][1])

        start_of_context = (
            len(self._dataset_reader._tokenizer.sequence_pair_start_tokens)
            + len(instance["metadata"]["question_tokens"])
            + len(self._dataset_reader._tokenizer.sequence_pair_mid_tokens)
        )

        answer_span = SpanField(
            start_of_context + span_start,
            start_of_context + span_end,
            instance["question_with_context"],
        )
        new_instance.add_field("answer_span", answer_span)
        return [new_instance]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        logger.warning(
            "This method is implemented only for use in interpret modules."
            "The predictor maps a question to multiple instances. "
            "Please use _json_to_instances instead for all non-interpret uses. "
        )
        return self._json_to_instances(json_dict, qid=-1)[0]

    def _json_to_instances(self, json_dict: JsonDict, qid: Optional[int] = None) -> List[Instance]:
        # We allow the passage / context to be specified with either key.
        # But we do it this way so that a 'KeyError: context' exception will be raised
        # when neither key is specified, since the 'context' key is the default and
        # the 'passage' key was only added to be compatible with the input for other
        # RC models.
        # if `qid` is `None`, it is updated using self._next_qid
        context = json_dict["passage"] if "passage" in json_dict else json_dict["context"]
        result: List[Instance] = []
        question_id = qid or self._next_qid

        for instance in self._dataset_reader.make_instances(
            qid=str(question_id),
            question=json_dict["question"],
            answers=[],
            context=context,
            first_answer_offset=None,
            is_training=False,
        ):
            self._dataset_reader.apply_token_indexers(instance)
            result.append(instance)
        if qid is None:
            self._next_qid += 1
        return result

    @overrides
    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        instances = []
        for json_dict in json_dicts:
            instances.extend(self._json_to_instances(json_dict))
        return instances

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        result = self.predict_batch_instance(instances)
        assert len(result) == len(inputs)
        return result

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)

        # group outputs with the same question id
        qid_to_output: Dict[str, Dict[str, Any]] = {}
        for instance, output in zip(instances, outputs):
            qid = instance["metadata"]["id"]
            output["id"] = qid
            output["context_tokens"] = instance["metadata"]["context_tokens"]
            if instance["metadata"]["answers"]:
                output["answers"] = instance["metadata"]["answers"]
            if qid in qid_to_output:
                old_output = qid_to_output[qid]
                if old_output["best_span_scores"] < output["best_span_scores"]:
                    qid_to_output[qid] = output
            else:
                qid_to_output[qid] = output

        return [sanitize(o) for o in qid_to_output.values()]
