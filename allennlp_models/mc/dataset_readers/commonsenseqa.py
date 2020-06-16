import logging
from typing import List

from allennlp.data import DatasetReader, Instance
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("commonsenseqa")
class CommonsenseQaReader(DatasetReader):
    """
    Reads the input data for the CommonsenseQA dataset (https://arxiv.org/abs/1811.00937).

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
        from allennlp.common.file_utils import json_lines_from_file

        for json in json_lines_from_file(file_path):
            choices = [
                (choice["label"], choice["text"])
                for choice in json["question"]["choices"]
            ]
            correct_choice = [
                i for i, (label, _) in enumerate(choices) if label == json["answerKey"]
            ][0]
            yield self.text_to_instance(
                json["id"],
                json["question"]["stem"],
                [c[1] for c in choices],
                correct_choice)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        id: str,
        question: str,
        alternatives: List[str],
        correct_alternative: int,
    ) -> Instance:
        # tokenize
        question = self._tokenizer.tokenize(question)
        sequences = []
        for alternative in alternatives:
            alternative = self._tokenizer.tokenize(alternative)
            length_for_question = self.length_limit - len(alternative) - self._tokenizer.num_special_tokens_for_pair()
            sequences.append(
                self._tokenizer.add_special_tokens(question[:length_for_question], alternative)
            )

        # make fields
        from allennlp.data.fields import TextField

        sequences = [
            TextField(sequence, self._token_indexers) for sequence in sequences
        ]
        if correct_alternative < 0 or correct_alternative >= len(sequences):
            raise ValueError("Alternative %d does not exist", correct_alternative)
        from allennlp.data.fields import ListField

        sequences = ListField(sequences)

        from allennlp.data.fields import IndexField

        from allennlp.data.fields import MetadataField
        return Instance(
            {
                "alternatives": sequences,
                "correct_alternative": IndexField(correct_alternative, sequences),
                "id": MetadataField(id)
            }
        )
