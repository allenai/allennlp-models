from typing import Optional, Dict
import logging
import csv

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("quora_paraphrase")
class QuoraParaphraseDatasetReader(DatasetReader):
    """
    Reads a file from the Quora Paraphrase dataset. The train/validation/test split of the data
    comes from the paper [Bilateral Multi-Perspective Matching for
    Natural Language Sentences](https://arxiv.org/abs/1702.03814) by Zhiguo Wang et al., 2017.
    Each file of the data is a tsv file without header. The columns are is_duplicate, question1,
    question2, and id.  All questions are pre-tokenized and tokens are space separated. We convert
    these keys into fields named "label", "premise" and "hypothesis", so that it is compatible t
    some existing natural language inference algorithms.

    Registered as a `DatasetReader` with name "quora_paraphrase".

    # Parameters

    tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the premise and hypothesis into words or other kinds of tokens.
        Defaults to `WhitespaceTokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input token representations. Defaults to `{"tokens":
        SingleIdTokenIndexer()}`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens

        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter="\t")
            for row in tsv_in:
                if len(row) == 4:
                    yield self.text_to_instance(premise=row[1], hypothesis=row[2], label=row[0])

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: str = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        premise = self._tokenizer.tokenize(premise)
        hypothesis = self._tokenizer.tokenize(hypothesis)

        if self._combine_input_fields:
            tokens = self._tokenizer.add_special_tokens(premise, hypothesis)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        else:
            premise_tokens = self._tokenizer.add_special_tokens(premise)
            hypothesis_tokens = self._tokenizer.add_special_tokens(hypothesis)
            fields["premise"] = TextField(premise_tokens, self._token_indexers)
            fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)

        if label is not None:
            fields["label"] = LabelField(label)

        return Instance(fields)
