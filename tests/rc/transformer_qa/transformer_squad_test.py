import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list

from allennlp_models.rc.transformer_qa import TransformerSquadReader
from tests import FIXTURES_ROOT


class TestTransformerSquadReader:
    def test_read_from_file(self):
        reader = TransformerSquadReader()
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))
        assert len(instances) == 5

        token_text = [t.text for t in instances[0].fields["question_with_context"].tokens]
        token_type_ids = [t.type_id for t in instances[0].fields["question_with_context"].tokens]

        assert token_text[:3] == ["[CLS]", "To", "whom"]
        assert token_type_ids[:3] == [1, 1, 1]

        assert token_text[-3:] == ["Mary", ".", "[SEP]"]
        assert token_type_ids[-3:] == [0, 0, 1]

        assert token_text[instances[0].fields["context_span"].span_start] == "Architectural"
        assert token_type_ids[instances[0].fields["context_span"].span_start] == 0

        assert token_text[instances[0].fields["context_span"].span_end + 1] == "[SEP]"
        assert token_type_ids[instances[0].fields["context_span"].span_end + 1] == 1
        assert token_text[instances[0].fields["context_span"].span_end] == "."
        assert token_type_ids[instances[0].fields["context_span"].span_end] == 0

        assert token_text[
            instances[0]
            .fields["answer_span"]
            .span_start : instances[0]
            .fields["answer_span"]
            .span_end
            + 1
        ] == ["Saint", "Bern", "##ade", "##tte", "So", "##ubi", "##rous"]

        for instance in instances:
            token_type_ids = [t.type_id for t in instance.fields["question_with_context"].tokens]
            context_start = instance.fields["context_span"].span_start
            context_end = instance.fields["context_span"].span_end + 1
            assert all(id != 0 for id in token_type_ids[:context_start])
            assert all(id == 0 for id in token_type_ids[context_start:context_end])
            assert all(id != 0 for id in token_type_ids[context_end:])

    def test_length_limit_works(self):
        max_query_length = 10
        stride = 20

        reader = TransformerSquadReader(
            length_limit=100,
            max_query_length=max_query_length,
            stride=stride,
            skip_invalid_examples=False,
        )
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))

        assert len(instances) == 12
        # The sequence is "<s> question </s> </s> context".
        assert instances[0].fields["context_span"].span_start == 1 + max_query_length + 2

        instance_0_text = [t.text for t in instances[0].fields["question_with_context"].tokens]
        instance_1_text = [t.text for t in instances[1].fields["question_with_context"].tokens]
        assert instance_0_text[: max_query_length + 2] == instance_1_text[: max_query_length + 2]
        assert instance_0_text[max_query_length + 3] != instance_1_text[max_query_length + 3]
        assert instance_0_text[-1] == "[SEP]"
        assert instance_0_text[-2] == "G"
        assert instance_1_text[instances[1].fields["context_span"].span_start + stride - 1] == "G"

    def test_roberta_bug(self):
        """This reader tokenizes first by spaces, and then re-tokenizes using the wordpiece tokenizer that comes
        with the transformer model. For RoBERTa, this produces a bug, since RoBERTa tokens are different depending
        on whether they are preceded by a space, and the first round of tokenization cuts off the spaces. The
        reader has a workaround for this case. This tests that workaround."""
        reader = TransformerSquadReader(transformer_model_name="roberta-base")
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))
        assert instances
        assert len(instances) == 5
        token_text = [t.text for t in instances[1].fields["question_with_context"].tokens]
        token_ids = [t.text_id for t in instances[1].fields["question_with_context"].tokens]

        assert token_text[:3] == ["<s>", "What", "Ġsits"]
        assert token_ids[:3] == [0, 2264, 6476]
