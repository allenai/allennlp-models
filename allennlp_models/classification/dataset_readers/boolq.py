import json
import logging
from typing import Optional, Iterable, Dict

from allennlp.common.file_utils import cached_path
from overrides import overrides
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Instance, Field
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField

logger = logging.getLogger(__name__)


@DatasetReader.register("boolq")
class BoolQDatasetReader(DatasetReader):
    """
    This DatasetReader is designed to read in the BoolQ data
    for binary QA task. It returns a dataset of instances with the
    following fields:
    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`
    Registered as a `DatasetReader` with name "boolq".
    # Parameters
    tokenizer: `Tokenizer`, optional (default=`WhitespaceTokenizer()`)
        Tokenizer to use to split the input sequences into words or other kinds of tokens.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    """

    def __init__(
        self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, **kwargs
    ):
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        file_path = cached_path(file_path, extract_archive=True)
        with open(file_path) as f:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in self.shard_iterable(f):
                record = json.loads(line.strip())
                yield self.text_to_instance(
                    passage=record.get("passage"),
                    question=record.get("question"),
                    label=record.get("label"),
                )

    @overrides
    def text_to_instance(  # type: ignore
        self, passage: str, question: str, label: Optional[bool] = None
    ) -> Instance:
        """
        We take the passage and the question as input, tokenize and concat them.
        # Parameters
        passage : `str`, required.
            The passage in a given BoolQ record.
        question : `str`, required.
            The passage in a given BoolQ record.
        label : `bool`, optional, (default = `None`).
            The label for the passage and the question.
        # Returns
        An `Instance` containing the following fields:
            tokens : `TextField`
                The tokens in the concatenation of the passage and the question.
            label : `LabelField`
                The answer to the question.
        """
        fields: Dict[str, Field] = {}

        # 80% of the question length in the training set is less than 60, 512 - 4 - 60 = 448.
        passage_tokens = self.tokenizer.tokenize(passage)[:448]
        question_tokens = self.tokenizer.tokenize(question)[:60]

        tokens = self.tokenizer.add_special_tokens(passage_tokens, question_tokens)
        text_field = TextField(tokens)
        fields["tokens"] = text_field

        if label is not None:
            label_field = LabelField(int(label), skip_indexing=True)
            fields["label"] = label_field
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"].token_indexers = self.token_indexers  # type: ignore
