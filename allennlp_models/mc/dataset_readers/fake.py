import logging
from typing import List

from allennlp.data import DatasetReader, Instance
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("fake")
class FakeReader(DatasetReader):
    """
    Creates fake multiple-choice input. If your model doesn't get 99% on this data, it is broken.

    Instances have two fields:
     * `alternatives`, a ListField of TextField
     * `correct_alternative`, IndexField with the correct answer among `alternatives`

    Parameters
    ----------
    transformer_model_name : `str`, optional (default=`roberta-large`)
        This reader chooses tokenizer and token indexer according to this setting.
    length_limit : `int`, optional (default=512)
        We will make sure that the length of the alternatives never exceeds this many word pieces.
    """

    def __init__(
        self, transformer_model_name: str = "roberta-large", length_limit: int = 512, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if self.max_instances is None:
            raise ValueError("FakeReader requires max_instances to be set.")

        from allennlp.data.tokenizers import PretrainedTransformerTokenizer

        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )

        from allennlp.data.token_indexers import PretrainedTransformerIndexer

        self._token_indexers = {"tokens": PretrainedTransformerIndexer(transformer_model_name)}
        self.length_limit = length_limit

    @overrides
    def _read(self, file_path: str):
        logger.info("Ignoring file at %s", file_path)

        for i in range(self.max_instances):
            label = i % 2
            texts = [f"This is the false choice {i}."] * 2
            texts[label] = f"This is the true choice {i}."
            yield self.text_to_instance(texts, label)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        alternatives: List[str],
        correct_alternative: int,
    ) -> Instance:
        # tokenize
        alternatives = [self._tokenizer.tokenize(alternative) for alternative in alternatives]

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
            raise ValueError("Alternative %d does not exist.", correct_alternative)
        from allennlp.data.fields import ListField

        alternatives = ListField(alternatives)

        from allennlp.data.fields import IndexField

        return Instance(
            {
                "alternatives": alternatives,
                "correct_alternative": IndexField(correct_alternative, alternatives),
            }
        )
