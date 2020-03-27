from typing import Dict, List, Optional
import logging
import copy

from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import IndexField, Field, ListField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("masked_language_modeling")
class MaskedLanguageModelingReader(DatasetReader):
    """
    Reads a text file and converts it into a `Dataset` suitable for training a masked language
    model.

    The :class:`Field` s that we create are the following: an input `TextField`, a mask position
    `ListField[IndexField]`, and a target token `TextField` (the target tokens aren't a single
    string of text, but we use a `TextField` so we can index the target tokens the same way as
    our input, typically with a single `PretrainedTransformerIndexer`).  The mask position and
    target token lists are the same length.

    NOTE: This is not fully functional!  It was written to put together a demo for interpreting and
    attacking masked language modeling, not for actually training anything.  `text_to_instance`
    is functional, but `_read` is not.  To make this fully functional, you would want some
    sampling strategies for picking the locations for [MASK] tokens, and probably a bunch of
    efficiency / multi-processing stuff.

    # Parameters

    model_name : `str`, optional (default=`"bert-base-cased"`)
        Name of the pretrained model.
    """

    def __init__(
        self, model_name: str = "bert-base-cased", **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = PretrainedTransformerTokenizer(model_name=model_name)
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name=model_name)
        }

    @overrides
    def _read(self, file_path: str):
        import sys

        # You can call pytest with either `pytest` or `py.test`.
        if "test" not in sys.argv[0]:
            logger.error("_read is only implemented for unit tests at the moment")
        mask_token = self._tokenizer.tokenize("[MASK]")[1]
        with open(file_path, "r") as text_file:
            for sentence in text_file:
                tokens = self._tokenizer.tokenize(sentence)
                target = tokens[1]
                tokens[1] = mask_token
                yield self.text_to_instance(sentence, tokens, [target])

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentence: Optional[str] = None,
        tokens: Optional[List[Token]] = None,
        targets: Optional[List[Token]] = None,
    ) -> Instance:

        """
        # Parameters

        sentence : `str`, optional
            A sentence containing [MASK] tokens that should be filled in by the model.  This input
            is superceded and ignored if `tokens` is given.
        tokens : `List[Token]`, optional
            An already-tokenized sentence containing some number of [MASK] tokens to be predicted.
        targets : `List[Token]`, optional
            Contains the target tokens to be predicted.  The length of this list should be the same
            as the number of [MASK] tokens in the input.
        """
        if tokens is None:
            tokens = self._tokenizer.tokenize(sentence)
        input_field = TextField(tokens, self._token_indexers)
        mask_positions = []
        for i, token in enumerate(tokens):
            if token.text == "[MASK]":
                mask_positions.append(i)
        if len(mask_positions) <= 0:
            raise ValueError("No [MASK] tokens found!")
        if targets is not None and len(targets) != len(mask_positions):
            raise ValueError(f"Found {len(mask_positions)} mask tokens and {len(targets)} targets")
        mask_position_field = ListField([IndexField(i, input_field) for i in mask_positions])
        fields: Dict[str, Field] = {
            "tokens": input_field,
            "mask_positions": mask_position_field,
        }
        if targets is not None:
            fields["target_ids"] = TextField(targets, self._token_indexers)

        return Instance(fields)
