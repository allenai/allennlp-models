import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable

from overrides import overrides

from allennlp.common.util import sanitize_wordpiece
from allennlp.data.fields import MetadataField, TextField, SpanField, IndexField
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

from allennlp_models.rc.dataset_readers.utils import char_span_to_token_span

logger = logging.getLogger(__name__)


@DatasetReader.register("transformer_squad")
class TransformerSquadReader(DatasetReader):
    """
    Dataset reader suitable for JSON-formatted SQuAD-like datasets to be used with a transformer-based
    QA model, such as [`TransformerQA`](../../models/transformer_qa#TransformerQA).

    It will generate `Instances` with the following fields:

     * `question_with_context`, a `TextField` that contains the concatenation of question and context,
     * `answer_span`, a `SpanField` into the `question` `TextField` denoting the answer.
     * `context_span`, a `SpanField` into the `question` `TextField` denoting the context, i.e., the part of
       the text that potential answers can come from.
     * `cls_index` (optional), an `IndexField` that holds the index of the `[CLS]` token within the
       `question_with_context` field. This is needed because the `[CLS]` token is used to indicate
       an impossible question. Since most tokenizers/models have the `[CLS]` token as the first
       token, this will only be included in the instance if the `[CLS]` token is NOT the first token.
     * `metadata`, a `MetadataField` that stores the instance's ID, the original question, the original
       passage text, both of these in tokenized form, and the gold answer strings, accessible as
       `metadata['id']`, `metadata['question']`, `metadata['context']`, `metadata['question_tokens']`,
       `metadata['context_tokens']`, and `metadata['answers']`. This is so that we can more easily use the
       official SQuAD evaluation script to get metrics.

    For SQuAD v2.0-style datasets that contain impossible questions, we set the gold answer span
    to the span of the `[CLS]` token when there are no answers.

    We also support limiting the maximum length for the question. When the context+question is too long, we run a
    sliding window over the context and emit multiple instances for a single question.
    If `skip_impossible_questions` is `True`, then we only emit instances that contain a gold answer.
    As a result, the per-instance metrics you get during training and evaluation might not correspond
    100% to the SQuAD task.

    To get a final number for SQuAD v1.1, you have to run

    ```
    python -m allennlp_models.rc.tools.transformer_qa_eval
    ```

    # Parameters

    transformer_model_name : `str`, optional (default=`'bert-base-cased'`)
        This reader chooses tokenizer and token indexer according to this setting.

    length_limit : `int`, optional (default=`384`)
        We will make sure that the length of context+question never exceeds this many word pieces.

    stride : `int`, optional (default=`128`)
        When context+question are too long for the length limit, we emit multiple instances for one question,
        where the context is shifted. This parameter specifies the overlap between the shifted context window. It
        is called "stride" instead of "overlap" because that's what it's called in the original huggingface
        implementation.

    skip_impossible_questions : `bool`, optional (default=`False`)
        If this is true, we will skip examples that don't have an answer. This could happen if the question
        is marked impossible in the dataset, or if the question+context is truncated according to `length_limit`
        such that the context no longer contains a gold answer.

        For SQuAD v1.1-style datasets, you should set this to `True` during training, and `False` any other time.

        For SQuAD v2.0-style datasets you should leave this as `False`.

    max_query_length : `int`, optional (default=`64`)
        The maximum number of wordpieces dedicated to the question. If the question is longer than this, it will be
        truncated.

    """

    def __init__(
        self,
        transformer_model_name: str = "bert-base-cased",
        length_limit: int = 384,
        stride: int = 128,
        skip_impossible_questions: bool = False,
        max_query_length: int = 64,
        tokenizer_kwargs: Dict[str, Any] = None,
        **kwargs
    ) -> None:
        if "skip_invalid_examples" in kwargs:
            import warnings

            warnings.warn(
                "'skip_invalid_examples' is deprecated, please use 'skip_impossible_questions' instead",
                DeprecationWarning,
            )
            skip_impossible_questions = kwargs.pop("skip_invalid_examples")

        super().__init__(**kwargs)
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
        self.length_limit = length_limit
        self.stride = stride
        self.skip_impossible_questions = skip_impossible_questions
        self.max_query_length = max_query_length
        self._cls_token = self._tokenizer.tokenizer.cls_token
        # We'll include the `cls_index` IndexField in instances if the CLS token is
        # not always the first token.
        self._include_cls_index = (
            self._find_cls_index(
                self._tokenizer.add_special_tokens(
                    self._tokenizer.tokenize("a"), self._tokenizer.tokenize("a")
                )
            )
            != 0
        )

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
        logger.info("Reading the dataset")
        yielded_question_count = 0
        questions_with_more_than_one_instance = 0
        for article in dataset:
            for paragraph_json in article["paragraphs"]:
                context = paragraph_json["context"]
                for question_answer in paragraph_json["qas"]:
                    answers = [answer_json["text"] for answer_json in question_answer["answers"]]

                    # Just like huggingface, we only use the first answer for training.
                    if len(answers) > 0:
                        first_answer_offset = int(question_answer["answers"][0]["answer_start"])
                    else:
                        first_answer_offset = None

                    instances = self.make_instances(
                        question_answer.get("id", None),
                        question_answer["question"],
                        answers,
                        context,
                        first_answer_offset=first_answer_offset,
                        always_add_answer_span=True,
                    )
                    instances_yielded = 0
                    for instance in instances:
                        yield instance
                        instances_yielded += 1
                    if instances_yielded > 1:
                        questions_with_more_than_one_instance += 1
                    yielded_question_count += 1

        if questions_with_more_than_one_instance > 0:
            logger.info(
                "%d (%.2f%%) questions have more than one instance",
                questions_with_more_than_one_instance,
                100 * questions_with_more_than_one_instance / yielded_question_count,
            )

    def make_instances(
        self,
        qid: str,
        question: str,
        answers: List[str],
        context: str,
        first_answer_offset: Optional[int],
        always_add_answer_span: bool = False,
    ) -> Iterable[Instance]:
        """
        Create training instances from a SQuAD example.
        """
        # tokenize context by spaces first, and then with the wordpiece tokenizer
        # For RoBERTa, this produces a bug where every token is marked as beginning-of-sentence. To fix it, we
        # detect whether a space comes before a word, and if so, add "a " in front of the word.
        def tokenize_slice(start: int, end: int) -> Iterable[Token]:
            text_to_tokenize = context[start:end]
            if start - 1 >= 0 and context[start - 1].isspace():
                prefix = "a "  # must end in a space, and be short so we can be sure it becomes only one token
                wordpieces = self._tokenizer.tokenize(prefix + text_to_tokenize)
                for wordpiece in wordpieces:
                    if wordpiece.idx is not None:
                        wordpiece.idx -= len(prefix)
                return wordpieces[1:]
            else:
                return self._tokenizer.tokenize(text_to_tokenize)

        tokenized_context = []
        token_start = 0
        for i, c in enumerate(context):
            if c.isspace():
                for wordpiece in tokenize_slice(token_start, i):
                    if wordpiece.idx is not None:
                        wordpiece.idx += token_start
                    tokenized_context.append(wordpiece)
                token_start = i + 1
        for wordpiece in tokenize_slice(token_start, len(context)):
            if wordpiece.idx is not None:
                wordpiece.idx += token_start
            tokenized_context.append(wordpiece)

        if first_answer_offset is None:
            (token_answer_span_start, token_answer_span_end) = (-1, -1)
        else:
            (token_answer_span_start, token_answer_span_end), _ = char_span_to_token_span(
                [
                    (t.idx, t.idx + len(sanitize_wordpiece(t.text))) if t.idx is not None else None
                    for t in tokenized_context
                ],
                (first_answer_offset, first_answer_offset + len(answers[0])),
            )

        # Tokenize the question.
        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_question = tokenized_question[: self.max_query_length]

        # Stride over the context, making instances.
        space_for_context = (
            self.length_limit
            - len(tokenized_question)
            - len(self._tokenizer.sequence_pair_start_tokens)
            - len(self._tokenizer.sequence_pair_mid_tokens)
            - len(self._tokenizer.sequence_pair_end_tokens)
        )
        stride_start = 0
        while True:
            tokenized_context_window = tokenized_context[stride_start:]
            tokenized_context_window = tokenized_context_window[:space_for_context]

            window_token_answer_span = (
                token_answer_span_start - stride_start,
                token_answer_span_end - stride_start,
            )
            if any(i < 0 or i >= len(tokenized_context_window) for i in window_token_answer_span):
                # The answer is not contained in the window.
                window_token_answer_span = None

            if not self.skip_impossible_questions or window_token_answer_span is not None:
                additional_metadata = {"id": qid}
                instance = self.text_to_instance(
                    question,
                    tokenized_question,
                    context,
                    tokenized_context_window,
                    answers=answers,
                    token_answer_span=window_token_answer_span,
                    additional_metadata=additional_metadata,
                    always_add_answer_span=always_add_answer_span,
                )
                yield instance

            stride_start += space_for_context
            if stride_start >= len(tokenized_context):
                break
            stride_start -= self.stride

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        tokenized_question: List[Token],
        context: str,
        tokenized_context: List[Token],
        answers: List[str] = None,
        token_answer_span: Optional[Tuple[int, int]] = None,
        additional_metadata: Dict[str, Any] = None,
        always_add_answer_span: bool = False,
    ) -> Instance:
        fields = {}

        # make the question field
        question_field = TextField(
            self._tokenizer.add_special_tokens(tokenized_question, tokenized_context),
            self._token_indexers,
        )
        fields["question_with_context"] = question_field

        cls_index = self._find_cls_index(question_field.tokens)
        if self._include_cls_index:
            fields["cls_index"] = IndexField(cls_index, question_field)

        start_of_context = (
            len(self._tokenizer.sequence_pair_start_tokens)
            + len(tokenized_question)
            + len(self._tokenizer.sequence_pair_mid_tokens)
        )

        # make the answer span
        if token_answer_span is not None:
            assert all(i >= 0 for i in token_answer_span)
            assert token_answer_span[0] <= token_answer_span[1]

            fields["answer_span"] = SpanField(
                token_answer_span[0] + start_of_context,
                token_answer_span[1] + start_of_context,
                question_field,
            )
        elif always_add_answer_span:
            fields["answer_span"] = SpanField(cls_index, cls_index, question_field)

        # make the context span, i.e., the span of text from which possible answers should be drawn
        fields["context_span"] = SpanField(
            start_of_context, start_of_context + len(tokenized_context) - 1, question_field
        )

        # make the metadata
        metadata = {
            "question": question,
            "question_tokens": tokenized_question,
            "context": context,
            "context_tokens": tokenized_context,
            "answers": answers or [],
        }
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def _find_cls_index(self, tokens: List[Token]) -> int:
        return next(i for i, t in enumerate(tokens) if t.text == self._cls_token)
