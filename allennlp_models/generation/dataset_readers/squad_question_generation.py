from typing import Iterable, Optional, List, Any, Dict
import json
import logging
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SPAN_START_TOKEN = "<m>"
SPAN_END_TOKEN = "</m>"
ALL_SPECIAL_TOKENS = [SPAN_START_TOKEN, SPAN_END_TOKEN]


def _KnuthMorrisPratt(text, pattern):
    # Knuth-Morris-Pratt string matching
    # David Eppstein, UC Irvine, 1 Mar 2002

    # from http://code.activestate.com/recipes/117214/
    """Yields all starting positions of copies of the pattern in the text.
    Calling conventions are similar to string.find, but its arguments can be
    lists or iterators, not just strings, it returns all matches, not just
    the first one, and it does not need the whole text in memory at once.
    Whenever it yields, it will have read the text exactly up to and including
    the match that caused the yield."""

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos - shift]:
            shift += shifts[pos - shift]
        shifts[pos + 1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos


@DatasetReader.register("squad_question_generation")
class SquadConditionalQuestionGenerationReader(DatasetReader):
    def __init__(self, model_name: str, max_instances=None, lazy: bool = False):
        super().__init__(lazy=lazy)
        # Setting this to false to encode pairs of text
        self.tokenizer = PretrainedTransformerTokenizer(model_name, add_special_tokens=False)
        self.token_indexers = {
            "tokens": PretrainedTransformerIndexer(model_name, namespace="tokens")
        }

        # Add the tokens which will mark the answer span
        self.tokenizer.tokenizer.add_tokens(ALL_SPECIAL_TOKENS)
        self.max_instances = max_instances

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            squad_dataset = json.load(dataset_file)
        logger.info("Reading the dataset")

        total_ques, instances_read = 0, 0

        squad_dataset = squad_dataset["data"]
        for article in squad_dataset:
            title = article["title"]
            for para_idx, paragraph_json in enumerate(article["paragraphs"]):
                passage_id = f"{title}_{para_idx}"
                passage = paragraph_json["context"]
                for question_answer in paragraph_json["qas"]:
                    total_ques += 1
                    question = question_answer["question"].strip().replace("\n", "")
                    query_id = question_answer["id"]
                    answer = question_answer["answers"][0]
                    answer_text = answer["text"]
                    if not answer_text:
                        continue

                    answer_start_charoffsets = list(_KnuthMorrisPratt(passage, answer_text))
                    if not answer_start_charoffsets:
                        continue

                    instances_read += 1
                    if self.max_instances is not None and instances_read > self.max_instances:
                        break

                    yield self.text_to_instance(
                        passage=passage,
                        answer_text=answer_text,
                        answer_start_charoffsets=answer_start_charoffsets,
                        passage_id=passage_id,
                        query_id=query_id,
                        question=question,
                    )
            if self.max_instances is not None and instances_read > self.max_instances:
                break

        logger.info("Total questions: {} Instances read: {}".format(total_ques, instances_read))

    def _insert_span_symbols(self, context: str, start: int, end: int) -> str:
        return f"{context[:start]}{SPAN_START_TOKEN} {context[start:end]} {SPAN_END_TOKEN}{context[end:]}"

    @overrides
    def text_to_instance(
        self,
        passage: str,
        answer_text: str,
        answer_start_charoffsets: List[int],
        passage_id: str = None,
        query_id: str = None,
        question: Optional[str] = None,
    ) -> Instance:
        fields = {}
        # Using just the first occurrence of the answer in the passage
        answer_start_charoffset = answer_start_charoffsets[0]
        ans_end_charoffset = answer_start_charoffset + len(answer_text)

        ans_marked_passage = self._insert_span_symbols(
            passage, answer_start_charoffset, ans_end_charoffset
        )
        ans_marked_passage_tokens: List[Token] = self.tokenizer.tokenize(ans_marked_passage)
        # source: paragraph
        source_tokens = self.tokenizer.add_special_tokens(ans_marked_passage_tokens)
        fields["source_tokens"] = TextField(source_tokens, self.token_indexers)

        metadata: Dict[str, Any] = {}
        metadata["answer"] = answer_text
        metadata["answer_start"] = answer_start_charoffset
        metadata["answer_end"] = ans_end_charoffset
        metadata["passage"] = passage
        metadata["ans_marked_passage"] = ans_marked_passage
        metadata["source_tokens"] = source_tokens

        if question is not None:
            target_tokens = self.tokenizer.tokenize(question)
            target_tokens = self.tokenizer.add_special_tokens(target_tokens)
            fields["target_tokens"] = TextField(target_tokens, self.token_indexers)
            metadata["question"] = question
            metadata["target_tokens"] = target_tokens

        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
