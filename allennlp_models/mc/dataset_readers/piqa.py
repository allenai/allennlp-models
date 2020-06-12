import logging
from typing import List

from allennlp.data import DatasetReader, Instance
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("piqa")
class PiqaReader(DatasetReader):
    """
    Reads the input data for the PIQA dataset (https://arxiv.org/abs/1911.11641).

    Instances have two fields:
     * ``alternatives``, a ListField of TextField
     * ``correct_alternative``, IndexField with the correct answer among ``alternatives``

    Parameters
    ----------
    transformer_model_name : ``str``, optional (default=``roberta-large``)
        This reader chooses tokenizer and token indexer according to this setting.
    length_limit : ``int``, optional (default=512)
        We will make sure that the length of context+question never exceeds this many word pieces.
    """

    def __init__(
        self, transformer_model_name: str = "roberta-large", length_limit: int = 512, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        from allennlp.data.tokenizers import PretrainedTransformerTokenizer

        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )
        from allennlp.data.token_indexers import PretrainedTransformerIndexer

        self._token_indexers = {"tokens": PretrainedTransformerIndexer(transformer_model_name)}
        self.length_limit = length_limit

    @overrides
    def _read(self, file_path: str):
        import re

        labels_path = re.sub(r"\.jsonl$", "-labels.lst", file_path, 1)
        if labels_path == file_path:
            raise ValueError(
                "Could not determine file name for the labels corresponding to %s.", file_path
            )

        from allennlp.common.file_utils import cached_path

        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        from allennlp.common.file_utils import json_lines_from_file

        json_lines = json_lines_from_file(file_path)

        labels_path = cached_path(labels_path)
        from allennlp.common.file_utils import text_lines_from_file

        logger.info("Reading labels at %s", labels_path)
        labels_lines = text_lines_from_file(labels_path)

        for json, label in zip(json_lines, labels_lines):
            goal = json["goal"]
            sol1 = json["sol1"]
            sol2 = json["sol2"]
            label = int(label)
            yield self.text_to_instance([goal + " " + sol1, goal + " " + sol2], label)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        alternatives: List[str],
        correct_alternative: int,
    ) -> Instance:
        # tokenize
        alternatives = [self._tokenizer.tokenize(alternative) for alternative in alternatives]

        # truncate
        # TODO: A cleverer way of truncating would be to find where the differences are between the sequences, and
        # truncate so the differences persist.
        alternatives = [
            alternative[: self.length_limit - self._tokenizer.num_special_tokens_for_sequence()]
            for alternative in alternatives
        ]

        # add special tokens
        alternatives = [
            self._tokenizer.add_special_tokens(alternative) for alternative in alternatives
        ]

        # make fields
        from allennlp.data.fields import TextField

        alternatives = [
            TextField(alternative, self._token_indexers) for alternative in alternatives
        ]
        if correct_alternative < 0 or correct_alternative >= len(alternatives):
            raise ValueError("Alternative %d does not exist", correct_alternative)
        from allennlp.data.fields import ListField

        alternatives = ListField(alternatives)

        from allennlp.data.fields import IndexField

        return Instance(
            {
                "alternatives": alternatives,
                "correct_alternative": IndexField(correct_alternative, alternatives),
            }
        )
