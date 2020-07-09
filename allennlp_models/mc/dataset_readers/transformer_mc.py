import logging
from typing import List, Optional

from allennlp.data import DatasetReader, Instance
from overrides import overrides

logger = logging.getLogger(__name__)


class TransformerMCReader(DatasetReader):
    """
    Read input data for the TransformerMC model. This is the base class for all readers that produce
    data for TransformerMC.

    Instances have two fields:
     * `alternatives`, a ListField of TextField
     * `correct_alternative`, IndexField with the correct answer among `alternatives`

    Parameters
    ----------
    transformer_model_name : `str`, optional (default=`roberta-large`)
        This reader chooses tokenizer and token indexer according to this setting.
    length_limit : `int`, optional (default=`512`)
        We will make sure that the length of an alternative never exceeds this many word pieces.
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
    def text_to_instance(
        self,  # type: ignore
        qid: str,
        start: str,
        alternatives: List[str],
        label: Optional[int],
    ) -> Instance:
        # tokenize
        start = self._tokenizer.tokenize(start)

        sequences = []
        for alternative in alternatives:
            alternative = self._tokenizer.tokenize(alternative)
            length_for_start = (
                self.length_limit - len(alternative) - self._tokenizer.num_special_tokens_for_pair()
            )
            if length_for_start < 0:
                # If the alternative is too long by itself, we take the beginning and add no tokens from the start.
                alternative = alternative[:length_for_start]
                length_for_start = 0
            sequences.append(
                self._tokenizer.add_special_tokens(start[:length_for_start], alternative)
            )

        # make fields
        from allennlp.data.fields import TextField

        sequences = [TextField(sequence, self._token_indexers) for sequence in sequences]
        from allennlp.data.fields import ListField

        sequences = ListField(sequences)

        from allennlp.data.fields import MetadataField

        fields = {
            "alternatives": sequences,
            "qid": MetadataField(qid),
        }

        if label is not None:
            if label < 0 or label >= len(sequences):
                raise ValueError("Alternative %d does not exist", label)
            from allennlp.data.fields import IndexField

            fields["correct_alternative"] = IndexField(label, sequences)

        return Instance(fields)
