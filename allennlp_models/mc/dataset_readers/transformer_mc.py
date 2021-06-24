import logging
from typing import List, Optional

import torch
from allennlp.common import cached_transformers
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TransformerTextField
from overrides import overrides

logger = logging.getLogger(__name__)


class TransformerMCReader(DatasetReader):
    """
    Read input data for the TransformerMC model. This is the base class for all readers that produce
    data for TransformerMC.

    Instances have two fields:
     * `alternatives`, a ListField of TransformerTextField
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
        self._tokenizer = cached_transformers.get_tokenizer(transformer_model_name)
        self.length_limit = length_limit

    @overrides
    def text_to_instance(
        self,  # type: ignore
        qid: str,
        start: str,
        alternatives: List[str],
        label: Optional[int] = None,
    ) -> Instance:
        start = start.strip()
        alternatives = [f"{start} {a.strip()}" for a in alternatives]

        tokenized = self._tokenizer(
            alternatives,
            truncation="longest_first",
            max_length=self.length_limit,
            return_attention_mask=False,
        )
        sequences = [
            TransformerTextField(
                torch.IntTensor(input_ids), padding_token_id=self._tokenizer.pad_token_id
            )
            for input_ids in tokenized["input_ids"]
        ]

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
