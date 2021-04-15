import logging
from typing import Any, Dict

from overrides import overrides

from allennlp.data.fields import MetadataField, TextField, LabelField
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("transformer_superglue_rte")
class TransformerSuperGlueRteReader(DatasetReader):
    """
    Dataset reader for the SuperGLUE Recognizing Textual Entailment task, to be used with a transformer
    model such as RoBERTa. The dataset is in the JSON Lines format.

    It will generate `Instances` with the following fields:

     * `tokens`, a `TextField` that contains the concatenation of premise and hypothesis,
     * `label`, a `LabelField` containing the label, if one exists.
     * `metadata`, a `MetadataField` that stores the instance's index in the file, the original premise,
       the original hypothesis, both of these in tokenized form, and the gold label, accessible as
       `metadata['index']`, `metadata['premise']`, `metadata['hypothesis']`, `metadata['tokens']`,
       and `metadata['label']`.

    # Parameters

    type : `str`, optional (default=`'roberta-base'`)
        This reader chooses tokenizer according to this setting.
    """

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
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(
                transformer_model_name, tokenizer_kwargs=tokenizer_kwargs, max_length=512
            )
        }

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path, extract_archive=True)

        logger.info("Reading file at %s", file_path)
        yielded_relation_count = 0
        from allennlp.common.file_utils import json_lines_from_file

        for relation in self.shard_iterable(json_lines_from_file(file_path)):
            premise = relation["premise"]
            hypothesis = relation["hypothesis"]
            if "label" in relation:
                label = relation["label"]
            else:
                label = None
            index = relation["idx"]

            # todo: see if we even need this to be in a separate method
            instance = self.text_to_instance(index, label, premise, hypothesis)

            yield instance
            yielded_relation_count += 1

    @overrides
    def text_to_instance(
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
        fields["tokens"] = TextField(premise_and_hypothesis)

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

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["tokens"].token_indexers = self._token_indexers
