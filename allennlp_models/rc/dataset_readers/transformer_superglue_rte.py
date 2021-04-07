import jsonlines
import logging
from typing import Any, Dict, Iterable

from overrides import overrides

from allennlp.data.fields import MetadataField, TextField, LabelField
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

from allennlp_models.rc.dataset_readers.utils import char_span_to_token_span

logger = logging.getLogger(__name__)

@DatasetReader.register("transformer_superglue_rte")
class TransformerSuperGlueRteReader(DatasetReader):
    # todo: fix comment
    """
    Dataset reader for the SuperGLUE Recognizing Textual Entailment task, to be used with a transformer
    model such as RoBERTa. The dataset is in the JSON Lines format.

    It will generate `Instances` with the following fields:

     * `tokens`, a `TextField` that contains the concatenation of premise and hypothesis,
     * `label`, a `LabelField` containing the label, if one exists.
     * `metadata`, a `MetadataField` that stores the instance's ID, the original question, the original
       passage text, both of these in tokenized form, and the gold answer strings, accessible as
       `metadata['id']`, `metadata['question']`, `metadata['context']`, `metadata['question_tokens']`,
       `metadata['context_tokens']`, and `metadata['answers']`. This is so that we can more easily use the
       official SQuAD evaluation script to get metrics.

    # Parameters

    transformer_model_name : `str`, optional (default=`'bert-base-cased'`)
        This reader chooses tokenizer and token indexer according to this setting.
    """

    # todo: fix init
    def __init__(
        self,
        transformer_model_name: str = "roberta-base",
        tokenizer_kwargs: Dict[str, Any] = None,
        **kwargs
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name,
            add_special_tokens=False,
            tokenizer_kwargs=tokenizer_kwargs,
        )

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        yielded_relation_count = 0
        for relation in self.shard_iterable(jsonlines.open(file_path)):
            premise = relation["premise"]
            hypothesis = relation["hypothesis"]
            if "label" in relation:
                label = relation["label"]
            else:
                label = None
            index = relation["idx"]

            # todo: see if we even need this to be in a separate method
            instance = self.make_instance(index, label, premise, hypothesis)

            yield instance
            yielded_relation_count += 1

    def make_instance(
        self,
        index: int,
        label: str,
        premise: str,
        hypothesis: str,
    ) -> Instance:
        tokenized_premise = self._tokenizer.tokenize(premise)
        tokenized_hypothesis = self._tokenizer.tokenize(hypothesis)

        fields = {}

        premise_and_hypothesis = TextField(
            self._tokenizer.add_special_tokens(tokenized_premise, tokenized_hypothesis),
        )
        fields["premise_and_hypothesis"] = TextField(premise_and_hypothesis)

        # make the metadata
        metadata = {
            "premise": premise,
            "premise_tokens": tokenized_premise,
            "hypothesis": hypothesis,
            "hypothesis_tokens": tokenized_hypothesis,
            "index": index,
        }
        if label:
            fields["label"] = LabelField(label)
            metadata["label"] = label

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
