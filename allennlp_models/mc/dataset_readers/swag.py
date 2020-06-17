import logging
from typing import List

from allennlp.data import DatasetReader, Instance
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("swag")
class SwagReader(DatasetReader):
    """
    Reads the input data for the SWAG dataset (https://arxiv.org/abs/1808.05326).

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
        from allennlp.common.file_utils import cached_path

        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            import csv

            for line_number, line in enumerate(csv.reader(f)):
                if line_number == 0:
                    continue

                yield self.text_to_instance(
                    id=line[1], start=line[3], alternatives=line[7:11], label=int(line[11])
                )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        id: str,
        start: str,
        alternatives: List[str],
        label: int,
    ) -> Instance:
        # tokenize
        start = self._tokenizer.tokenize(start)

        sequences = []
        for alternative in alternatives:
            alternative = self._tokenizer.tokenize(alternative)
            length_for_start = (
                self.length_limit - len(alternative) - self._tokenizer.num_special_tokens_for_pair()
            )
            sequences.append(
                self._tokenizer.add_special_tokens(start[:length_for_start], alternative)
            )

        # make fields
        from allennlp.data.fields import TextField

        sequences = [TextField(sequence, self._token_indexers) for sequence in sequences]
        if label < 0 or label >= len(sequences):
            raise ValueError("Alternative %d does not exist", label)
        from allennlp.data.fields import ListField

        sequences = ListField(sequences)

        from allennlp.data.fields import IndexField

        from allennlp.data.fields import MetadataField

        return Instance(
            {
                "alternatives": sequences,
                "correct_alternative": IndexField(label, sequences),
                "id": MetadataField(id),
            }
        )
