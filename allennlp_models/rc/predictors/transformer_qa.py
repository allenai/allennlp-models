from typing import List, Dict, Any

from allennlp.models import Model
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.predictors.predictor import Predictor


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
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raise NotImplementedError(
            "This predictor maps a question to multiple instances. "
            "Please use _json_to_instances instead."
        )

    def _json_to_instances(self, json_dict: JsonDict) -> List[Instance]:
        # We allow the passage / context to be specified with either key.
        # But we do it this way so that a 'KeyError: context' exception will be raised
        # when neither key is specified, since the 'context' key is the default and
        # the 'passage' key was only added to be compatible with the input for other
        # RC models.
        context = json_dict["passage"] if "passage" in json_dict else json_dict["context"]
        result = list(
            self._dataset_reader.make_instances(
                qid=str(self._next_qid),
                question=json_dict["question"],
                answers=[],
                context=context,
                first_answer_offset=None,
            )
        )
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
