"""
Dataset reader for SuperGLUE's Reading Comprehension with Commonsense Reasoning task (Zhang Et
al. 2018).

Reader Implemented by Gabriel Orlanski
"""
import logging
from typing import Dict, List, Optional, Iterable, Union, Tuple, Any
from pathlib import Path
from allennlp.common.util import sanitize_wordpiece

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, SpanField
from allennlp.data.instance import Instance
from allennlp_models.rc.dataset_readers.utils import char_span_to_token_span
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
import json

logger = logging.getLogger(__name__)

__all__ = ["RecordTaskReader"]


# TODO: Optimize this reader


@DatasetReader.register("superglue_record")
class RecordTaskReader(DatasetReader):
    """
    Reader for Reading Comprehension with Commonsense Reasoning(ReCoRD) task from SuperGLUE. The
    task is detailed in the paper ReCoRD: Bridging the Gap between Human and Machine Commonsense
    Reading Comprehension (arxiv.org/pdf/1810.12885.pdf) by Zhang et al. Leaderboards and the
    official evaluation script for the ReCoRD task can be found sheng-z.github.io/ReCoRD-explorer/.

    The reader reads a JSON file in the format from
    sheng-z.github.io/ReCoRD-explorer/dataset-readme.txt


    # Parameters

    tokenizer: `Tokenizer`, optional
        The tokenizer class to use. Defaults to SpacyTokenizer

    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.

    passage_length_limit : `int`, optional (default=`None`)
        If specified, we will cut the passage if the length of passage exceeds this limit.

    question_length_limit : `int`, optional (default=`None`)
        If specified, we will cut the question if the length of question exceeds this limit.

    raise_errors: `bool`, optional (default=`False`)
        If the reader should raise errors or just continue.

    kwargs: `Dict`
        Keyword arguments to be passed to the DatasetReader parent class constructor.

    """

    def __init__(
        self,
        transformer_model_name: str = "bert-base-cased",
        length_limit: int = 384,
        question_length_limit: int = 64,
        stride: int = 128,
        raise_errors: bool = False,
        tokenizer_kwargs: Dict[str, Any] = None,
        one_instance_per_query: bool = False,
        max_instances: int = None,
        **kwargs,
    ) -> None:
        """
        Initialize the RecordTaskReader.
        """
        super(RecordTaskReader, self).__init__(
            manual_distributed_sharding=True, max_instances=max_instances, **kwargs
        )

        self._kwargs = kwargs

        self._model_name = transformer_model_name
        self._tokenizer_kwargs = tokenizer_kwargs or {}
        # Save the values passed to __init__ to protected attributes
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name,
            add_special_tokens=False,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(
                transformer_model_name, tokenizer_kwargs=tokenizer_kwargs
            )
        }
        self._length_limit = length_limit
        self._query_len_limit = question_length_limit
        self._stride = stride
        self._raise_errors = raise_errors
        self._cls_token = "@placeholder"
        self._one_instance_per_query = one_instance_per_query

    def _to_params(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for this class.

        # Returns

        `Dict[str, Any]` The config dict.
        """
        return {
            "type": "superglue_record",
            "transformer_model_name": self._model_name,
            "length_limit": self._length_limit,
            "question_length_limit": self._query_len_limit,
            "stride": self._stride,
            "raise_errors": self._raise_errors,
            "tokenizer_kwargs": self._tokenizer_kwargs,
            "one_instance_per_query": self._one_instance_per_query,
            "max_instances": self.max_instances,
            **self._kwargs,
        }

    def _read(self, file_path: Union[Path, str]) -> Iterable[Instance]:
        # IF `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        # Read the 'data' key from the dataset
        logger.info(f"Reading '{file_path}'")
        with open(file_path) as fp:
            dataset = json.load(fp)["data"]
        logger.info(f"Found {len(dataset)} examples from '{file_path}'")

        # Keep track of certain stats while reading the file
        # examples_multiple_instance_count: The number of questions with more than
        #   one instance. Can happen because there is multiple queries for a
        #   single passage.
        # passages_yielded: The total number of instances found/yielded.
        examples_multiple_instance_count = 0
        examples_no_instance_count = 0
        passages_yielded = 0

        # Iterate through every example from the ReCoRD data file.
        for example in dataset:

            # Get the list of instances for the current example
            instances_for_example = self.get_instances_from_example(example)

            # Keep track of number of instances for this specific example that
            # have been yielded. Since it instances_for_example is a generator, we
            # do not know its length. To address this, we create an counter int.
            instance_count = 0

            # Iterate through the instances and yield them.
            for instance in instances_for_example:
                yield instance
                instance_count += 1

            if instance_count == 0:
                logger.warning(f"Example '{example['id']}' had no instances.")
                examples_no_instance_count += 1

            # Check if there was more than one instance for this example. If
            # there was we increase examples_multiple_instance_count by 1.
            # Otherwise we increase by 0.
            examples_multiple_instance_count += 1 if instance_count > 1 else 0

            passages_yielded += instance_count

            # Check to see if we are over the max_instances to yield.
            if self.max_instances and passages_yielded > self.max_instances:
                logger.info("Passed max instances")
                break

        # Log pertinent information.
        if passages_yielded:
            logger.info(
                f"{examples_multiple_instance_count}/{passages_yielded} "
                f"({examples_multiple_instance_count / passages_yielded * 100:.2f}%) "
                f"examples had more than one instance"
            )
            logger.info(
                f"{examples_no_instance_count}/{passages_yielded} "
                f"({examples_no_instance_count / passages_yielded * 100:.2f}%) "
                f"examples had no instances"
            )
        else:
            logger.warning(f"Could not find any instances in '{file_path}'")

    def get_instances_from_example(
        self, example: Dict, always_add_answer_span: bool = False
    ) -> Iterable[Instance]:
        """
        Helper function to get instances from an example.

        Much of this comes from `transformer_squad.make_instances`

        # Parameters

        example: `Dict[str,Any]`
            The example dict.

        # Returns:

        `Iterable[Instance]` The instances for each example
        """
        # Get the passage dict from the example, it has text and
        # entities
        example_id: str = example["id"]
        passage_dict: Dict = example["passage"]
        passage_text: str = passage_dict["text"]

        # Tokenize the passage
        tokenized_passage: List[Token] = self.tokenize_str(passage_text)

        # TODO: Determine what to do with entities. Superglue marks them
        #   explicitly as input (https://arxiv.org/pdf/1905.00537.pdf)

        # Get the queries from the example dict
        queries: List = example["qas"]
        logger.debug(f"{len(queries)} queries for example {example_id}")

        # Tokenize and get the context windows for each queries
        for query in queries:

            # Create the additional metadata dict that will be passed w/ extra
            # data for each query. We store the question & query ids, all
            # answers, and other data following `transformer_qa`.
            additional_metadata = {
                "id": query["id"],
                "example_id": example_id,
            }
            instances_yielded = 0
            # Tokenize, and truncate, the query based on the max set in
            # `__init__`
            tokenized_query = self.tokenize_str(query["query"])[: self._query_len_limit]

            # Calculate where the context needs to start and how many tokens we have
            # for it. This is due to the limit on the number of tokens that a
            # transformer can use because they have quadratic memory usage. But if
            # you are reading this code, you probably know that.
            space_for_context = (
                self._length_limit
                - len(list(tokenized_query))
                # Used getattr so I can test without having to load a
                # transformer model.
                - len(getattr(self._tokenizer, "sequence_pair_start_tokens", []))
                - len(getattr(self._tokenizer, "sequence_pair_mid_tokens", []))
                - len(getattr(self._tokenizer, "sequence_pair_end_tokens", []))
            )

            # Check if answers exist for this query. We assume that there are no
            # answers for this query, and set the start and end index for the
            # answer span to -1.
            answers = query.get("answers", [])
            if not answers:
                logger.warning(f"Skipping {query['id']}, no answers")
                continue

            # Create the arguments needed for `char_span_to_token_span`
            token_offsets = [
                (t.idx, t.idx + len(sanitize_wordpiece(t.text))) if t.idx is not None else None
                for t in tokenized_passage
            ]

            # Get the token offsets for the answers for this current passage.
            answer_token_start, answer_token_end = (-1, -1)
            for answer in answers:

                # Try to find the offsets.
                offsets, _ = char_span_to_token_span(
                    token_offsets, (answer["start"], answer["end"])
                )

                # If offsets for an answer were found, it means the answer is in
                # the passage, and thus we can stop looking.
                if offsets != (-1, -1):
                    answer_token_start, answer_token_end = offsets
                    break

            # Go through the context and find the window that has the answer in it.
            stride_start = 0

            while True:
                tokenized_context_window = tokenized_passage[stride_start:]
                tokenized_context_window = tokenized_context_window[:space_for_context]

                # Get the token offsets w.r.t the current window.
                window_token_answer_span = (
                    answer_token_start - stride_start,
                    answer_token_end - stride_start,
                )
                if any(
                    i < 0 or i >= len(tokenized_context_window) for i in window_token_answer_span
                ):
                    # The answer is not contained in the window.
                    window_token_answer_span = None

                if (
                    # not self.skip_impossible_questions
                    window_token_answer_span
                    is not None
                ):
                    # The answer WAS found in the context window, and thus we
                    # can make an instance for the answer.
                    instance = self.text_to_instance(
                        query["query"],
                        tokenized_query,
                        passage_text,
                        tokenized_context_window,
                        answers=[answer["text"] for answer in answers],
                        token_answer_span=window_token_answer_span,
                        additional_metadata=additional_metadata,
                        always_add_answer_span=always_add_answer_span,
                    )
                    yield instance
                    instances_yielded += 1

                if instances_yielded == 1 and self._one_instance_per_query:
                    break

                stride_start += space_for_context

                # If we have reached the end of the passage, stop.
                if stride_start >= len(tokenized_passage):
                    break

                # I am not sure what this does...but it is here?
                stride_start -= self._stride

    def tokenize_slice(self, text: str, start: int = None, end: int = None) -> Iterable[Token]:
        """
        Get + tokenize a span from a source text.

        *Originally from the `transformer_squad.py`*

        # Parameters

        text: `str`
            The text to draw from.
        start: `int`
            The start index for the span.
        end: `int`
            The end index for the span. Assumed that this is inclusive.

        # Returns

        `Iterable[Token]` List of tokens for the retrieved span.
        """
        start = start or 0
        end = end or len(text)
        text_to_tokenize = text[start:end]

        # Check if this is the start of the text. If the start is >= 0, check
        # for a preceding space. If it exists, then we need to tokenize a
        # special way because of a bug with RoBERTa tokenizer.
        if start - 1 >= 0 and text[start - 1].isspace():

            # Per the original tokenize_slice function, you need to add a
            # garbage token before the actual text you want to tokenize so that
            # the tokenizer does not add a beginning of sentence token.
            prefix = "a "

            # Tokenize the combined prefix and text
            wordpieces = self._tokenizer.tokenize(prefix + text_to_tokenize)

            # Go through each wordpiece in the tokenized wordpieces.
            for wordpiece in wordpieces:

                # Because we added the garbage prefix before tokenize, we need
                # to adjust the idx such that it accounts for this. Therefore we
                # subtract the length of the prefix from each token's idx.
                if wordpiece.idx is not None:
                    wordpiece.idx -= len(prefix)

            # We do not want the garbage token, so we return all but the first
            # token.
            return wordpieces[1:]
        else:

            # Do not need any sort of prefix, so just return all of the tokens.
            return self._tokenizer.tokenize(text_to_tokenize)

    def tokenize_str(self, text: str) -> List[Token]:
        """
        Helper method to tokenize a string.

        Adapted from the `transformer_squad.make_instances`

        # Parameters
            text: `str`
                The string to tokenize.

        # Returns

        `Iterable[Tokens]` The resulting tokens.

        """
        # We need to keep track of the current token index so that we can update
        # the results from self.tokenize_slice such that they reflect their
        # actual position in the string rather than their position in the slice
        # passed to tokenize_slice. Also used to construct the slice.
        token_index = 0

        # Create the output list (can be any iterable) that will store the
        # tokens we found.
        tokenized_str = []

        # Helper function to update the `idx` and add every wordpiece in the
        # `tokenized_slice` to the `tokenized_str`.
        def add_wordpieces(tokenized_slice: Iterable[Token]) -> None:
            for wordpiece in tokenized_slice:
                if wordpiece.idx is not None:
                    wordpiece.idx += token_index
                tokenized_str.append(wordpiece)

        # Iterate through every character and their respective index in the text
        # to create the slices to tokenize.
        for i, c in enumerate(text):

            # Check if the current character is a space. If it is, we tokenize
            # the slice of `text` from `token_index` to `i`.
            if c.isspace():
                add_wordpieces(self.tokenize_slice(text, token_index, i))
                token_index = i + 1

        # Add the end slice that is not collected by the for loop.
        add_wordpieces(self.tokenize_slice(text, token_index, len(text)))

        return tokenized_str

    @staticmethod
    def get_spans_from_text(text: str, spans: List[Tuple[int, int]]) -> List[str]:
        """
        Helper function to get a span from a string

        # Parameter

        text: `str`
            The source string
        spans: `List[Tuple[int,int]]`
            List of start and end indices for spans.

            Assumes that the end index is inclusive. Therefore, for start
            index `i` and end index `j`, retrieves the span at `text[i:j+1]`.

        # Returns

        `List[str]` The extracted string from text.
        """
        return [text[start : end + 1] for start, end in spans]

    def text_to_instance(
        self,
        query: str,
        tokenized_query: List[Token],
        passage: str,
        tokenized_passage: List[Token],
        answers: List[str],
        token_answer_span: Optional[Tuple[int, int]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        always_add_answer_span: Optional[bool] = False,
    ) -> Instance:
        """
        A lot of this comes directly from the `transformer_squad.text_to_instance`
        """
        fields = {}

        # Create the query field from the tokenized question and context. Use
        # `self._tokenizer.add_special_tokens` function to add the necessary
        # special tokens to the query.
        query_field = TextField(
            self._tokenizer.add_special_tokens(
                # The `add_special_tokens` function automatically adds in the
                # separation token to mark the separation between the two lists of
                # tokens. Therefore, we can create the query field WITH context
                # through passing them both as arguments.
                tokenized_query,
                tokenized_passage,
            ),
            self._token_indexers,
        )

        # Add the query field to the fields dict that will be outputted as an
        # instance. Do it here rather than assign above so that we can use
        # attributes from `query_field` rather than continuously indexing
        # `fields`.
        fields["question_with_context"] = query_field

        # Calculate the index that marks the start of the context.
        start_of_context = (
            +len(tokenized_query)
            # Used getattr so I can test without having to load a
            # transformer model.
            + len(getattr(self._tokenizer, "sequence_pair_start_tokens", []))
            + len(getattr(self._tokenizer, "sequence_pair_mid_tokens", []))
        )

        # make the answer span
        if token_answer_span is not None:
            assert all(i >= 0 for i in token_answer_span)
            assert token_answer_span[0] <= token_answer_span[1]

            fields["answer_span"] = SpanField(
                token_answer_span[0] + start_of_context,
                token_answer_span[1] + start_of_context,
                query_field,
            )
        # make the context span, i.e., the span of text from which possible
        # answers should be drawn
        fields["context_span"] = SpanField(
            start_of_context, start_of_context + len(tokenized_passage) - 1, query_field
        )

        # make the metadata
        metadata = {
            "question": query,
            "question_tokens": tokenized_query,
            "context": passage,
            "context_tokens": tokenized_passage,
            "answers": answers or [],
        }
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def _find_cls_index(self, tokens: List[Token]) -> int:
        """
        From transformer_squad
        Args:
            self:
            tokens:

        Returns:

        """
        return next(i for i, t in enumerate(tokens) if t.text == self._cls_token)
