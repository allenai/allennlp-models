import json
import logging
from typing import Any, Dict, List, Tuple, Optional

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer

from allennlp_models.rc.dataset_readers import utils

logger = logging.getLogger(__name__)

SQUAD2_NO_ANSWER_TOKEN = "@@<NO_ANSWER>@@"
"""
The default `no_answer_token` for the [`squad2`](#squad2) reader.
"""


@DatasetReader.register("squad")
class SquadReader(DatasetReader):
    """
    !!! Note
        If you're training on SQuAD v1.1 you should use the [`squad1()`](#squad1) classmethod
        to instantiate this reader, and for SQuAD v2.0 you should use the
        [`squad2()`](#squad2) classmethod.

        Also, for transformer-based models you should be using the
        [`TransformerSquadReader`](../transformer_squad#transformersquadreader).

    Dataset reader suitable for JSON-formatted SQuAD-like datasets.
    It will generate `Instances` with the following fields:

      - `question`, a `TextField`,
      - `passage`, another `TextField`,
      - `span_start` and `span_end`, both `IndexFields` into the `passage` `TextField`,
      - and `metadata`, a `MetadataField` that stores the instance's ID, the original passage text,
        gold answer strings, and token offsets into the original passage, accessible as `metadata['id']`,
        `metadata['original_passage']`, `metadata['answer_texts']` and
        `metadata['token_offsets']`, respectively. This is so that we can more easily use the official
        SQuAD evaluation scripts to get metrics.

    We also support limiting the maximum length for both passage and question. However, some gold
    answer spans may exceed the maximum passage length, which will cause error in making instances.
    We simply skip these spans to avoid errors. If all of the gold answer spans of an example
    are skipped, during training, we will skip this example. During validating or testing, since
    we cannot skip examples, we use the last token as the pseudo gold answer span instead. The
    computed loss will not be accurate as a result. But this will not affect the answer evaluation,
    because we keep all the original gold answer texts.

    # Parameters

    tokenizer : `Tokenizer`, optional (default=`SpacyTokenizer()`)
        We use this `Tokenizer` for both the question and the passage.  See :class:`Tokenizer`.
        Default is `SpacyTokenizer()`.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    passage_length_limit : `int`, optional (default=`None`)
        If specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : `int`, optional (default=`None`)
        If specified, we will cut the question if the length of question exceeds this limit.
    skip_impossible_questions: `bool`, optional (default=`False`)
        If this is true, we will skip examples with questions that don't contain the answer spans.
    no_answer_token: `Optional[str]`, optional (default=`None`)
        A special token to append to each context. If using a SQuAD 2.0-style dataset, this
        should be set, otherwise an exception will be raised if an impossible question is
        encountered.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        passage_length_limit: int = None,
        question_length_limit: int = None,
        skip_impossible_questions: bool = False,
        no_answer_token: Optional[str] = None,
        **kwargs,
    ) -> None:
        if "skip_invalid_examples" in kwargs:
            import warnings

            warnings.warn(
                "'skip_invalid_examples' is deprecated, please use 'skip_impossible_questions' instead",
                DeprecationWarning,
            )
            skip_impossible_questions = kwargs.pop("skip_invalid_examples")

        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_impossible_questions = skip_impossible_questions
        self.no_answer_token = no_answer_token

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article["paragraphs"]:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                for question_answer in paragraph_json["qas"]:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    is_impossible = question_answer.get("is_impossible", False)
                    if is_impossible:
                        answer_texts: List[str] = []
                        span_starts: List[int] = []
                        span_ends: List[int] = []
                    else:
                        answer_texts = [answer["text"] for answer in question_answer["answers"]]
                        span_starts = [
                            answer["answer_start"] for answer in question_answer["answers"]
                        ]
                        span_ends = [
                            start + len(answer) for start, answer in zip(span_starts, answer_texts)
                        ]
                    additional_metadata = {"id": question_answer.get("id", None)}
                    instance = self.text_to_instance(
                        question_text,
                        paragraph,
                        is_impossible=is_impossible,
                        char_spans=zip(span_starts, span_ends),
                        answer_texts=answer_texts,
                        passage_tokens=tokenized_paragraph,
                        additional_metadata=additional_metadata,
                    )
                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question_text: str,
        passage_text: str,
        is_impossible: bool = None,
        char_spans: List[Tuple[int, int]] = None,
        answer_texts: List[str] = None,
        passage_tokens: List[Token] = None,
        additional_metadata: Dict[str, Any] = None,
    ) -> Optional[Instance]:
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)

        if self.no_answer_token is not None:
            if self.passage_length_limit is not None:
                passage_tokens = passage_tokens[: self.passage_length_limit - 1]
            passage_tokens = passage_tokens + [
                Token(
                    text=self.no_answer_token,
                    idx=passage_tokens[-1].idx + len(passage_tokens[-1].text) + 1,  # type: ignore
                    lemma_=self.no_answer_token,
                )
            ]
        elif self.passage_length_limit is not None:
            passage_tokens = passage_tokens[: self.passage_length_limit]

        question_tokens = self._tokenizer.tokenize(question_text)
        if self.question_length_limit is not None:
            question_tokens = question_tokens[: self.question_length_limit]

        if is_impossible:
            if self.no_answer_token is None:
                raise ValueError(
                    "This is a SQuAD 2.0 dataset, yet your using a SQuAD reader has 'no_answer_token' "
                    "set to `None`. "
                    "Consider specifying the 'no_answer_token' or using the 'squad2' reader instead, which "
                    f"by default uses '{SQUAD2_NO_ANSWER_TOKEN}' as the 'no_answer_token'."
                )
            answer_texts = [self.no_answer_token]
            token_spans: List[Tuple[int, int]] = [
                (len(passage_tokens) - 1, len(passage_tokens) - 1)
            ]
        else:
            char_spans = char_spans or []
            # We need to convert character indices in `passage_text` to token indices in
            # `passage_tokens`, as the latter is what we'll actually use for supervision.
            token_spans = []
            passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
            for char_span_start, char_span_end in char_spans:
                if char_span_end > passage_offsets[-1][1]:
                    continue
                (span_start, span_end), error = utils.char_span_to_token_span(
                    passage_offsets, (char_span_start, char_span_end)
                )
                if error:
                    logger.debug("Passage: %s", passage_text)
                    logger.debug("Passage tokens (with no-answer): %s", passage_tokens)
                    logger.debug("Question text: %s", question_text)
                    logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                    logger.debug("Token span: (%d, %d)", span_start, span_end)
                    logger.debug(
                        "Tokens in answer: %s",
                        passage_tokens[span_start : span_end + 1],
                    )
                    logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
                token_spans.append((span_start, span_end))
            # The original answer is filtered out
            if char_spans and not token_spans:
                if self.skip_impossible_questions:
                    return None
                else:
                    if self.no_answer_token is not None:
                        answer_texts = [self.no_answer_token]
                    token_spans.append(
                        (
                            len(passage_tokens) - 1,
                            len(passage_tokens) - 1,
                        )
                    )
        return utils.make_reading_comprehension_instance(
            question_tokens,
            passage_tokens,
            self._token_indexers,
            passage_text,
            token_spans,
            answer_texts,
            additional_metadata,
        )

    @classmethod
    def squad1(
        cls,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        passage_length_limit: int = None,
        question_length_limit: int = None,
        skip_impossible_questions: bool = False,
        **kwargs,
    ) -> "SquadReader":
        """
        Gives a `SquadReader` suitable for SQuAD v1.1.
        """
        return cls(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            passage_length_limit=passage_length_limit,
            question_length_limit=question_length_limit,
            skip_impossible_questions=skip_impossible_questions,
            no_answer_token=None,
            **kwargs,
        )

    @classmethod
    def squad2(
        cls,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        passage_length_limit: int = None,
        question_length_limit: int = None,
        skip_impossible_questions: bool = False,
        no_answer_token: str = SQUAD2_NO_ANSWER_TOKEN,
        **kwargs,
    ) -> "SquadReader":
        """
        Gives a `SquadReader` suitable for SQuAD v2.0.
        """
        return cls(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            passage_length_limit=passage_length_limit,
            question_length_limit=question_length_limit,
            skip_impossible_questions=skip_impossible_questions,
            no_answer_token=no_answer_token,
            **kwargs,
        )


DatasetReader.register("squad1", constructor="squad1")(SquadReader)
DatasetReader.register("squad2", constructor="squad2")(SquadReader)
