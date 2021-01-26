from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data import DatasetReader
import pytest

from allennlp_models.rc import SquadReader
from allennlp_models.rc.dataset_readers.squad import SQUAD2_NO_ANSWER_TOKEN
from tests import FIXTURES_ROOT


class TestSquadReader:
    def test_from_params(self):
        squad1_reader = DatasetReader.from_params(Params({"type": "squad1"}))
        assert squad1_reader.no_answer_token is None
        squad2_reader = DatasetReader.from_params(Params({"type": "squad2"}))
        assert squad2_reader.no_answer_token is not None
        with pytest.warns(DeprecationWarning):
            squad_reader = DatasetReader.from_params(
                Params({"type": "squad1", "skip_invalid_examples": True})
            )
            assert squad_reader.skip_impossible_questions is True

    def test_read_from_file(self):
        reader = SquadReader()
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))
        assert len(instances) == 5

        assert [t.text for t in instances[0].fields["question"].tokens[:3]] == ["To", "whom", "did"]
        assert [t.text for t in instances[0].fields["passage"].tokens[:3]] == [
            "Architecturally",
            ",",
            "the",
        ]
        assert [t.text for t in instances[0].fields["passage"].tokens[-3:]] == ["of", "Mary", "."]
        assert instances[0].fields["span_start"].sequence_index == 102
        assert instances[0].fields["span_end"].sequence_index == 104

        assert [t.text for t in instances[1].fields["question"].tokens[:3]] == [
            "What",
            "sits",
            "on",
        ]
        assert [t.text for t in instances[1].fields["passage"].tokens[:3]] == [
            "Architecturally",
            ",",
            "the",
        ]
        assert [t.text for t in instances[1].fields["passage"].tokens[-3:]] == ["of", "Mary", "."]
        assert instances[1].fields["span_start"].sequence_index == 17
        assert instances[1].fields["span_end"].sequence_index == 23

        # We're checking this case because I changed the answer text to only have a partial
        # annotation for the last token, which happens occasionally in the training data.  We're
        # making sure we get a reasonable output in that case here.
        assert [t.text for t in instances[3].fields["question"].tokens[:3]] == [
            "Which",
            "individual",
            "worked",
        ]
        assert [t.text for t in instances[3].fields["passage"].tokens[:3]] == ["In", "1882", ","]
        assert [t.text for t in instances[3].fields["passage"].tokens[-3:]] == [
            "Nuclear",
            "Astrophysics",
            ".",
        ]
        span_start = instances[3].fields["span_start"].sequence_index
        span_end = instances[3].fields["span_end"].sequence_index
        answer_tokens = instances[3].fields["passage"].tokens[span_start : (span_end + 1)]
        expected_answer_tokens = ["Father", "Julius", "Nieuwland"]
        assert [t.text for t in answer_tokens] == expected_answer_tokens

    def test_can_build_from_params(self):
        reader = SquadReader.from_params(Params({}))

        assert reader._tokenizer.__class__.__name__ == "SpacyTokenizer"
        assert reader._token_indexers["tokens"].__class__.__name__ == "SingleIdTokenIndexer"

    def test_length_limit_works(self):
        # We're making sure the length of the text is correct if length limit is provided.
        reader = SquadReader(
            passage_length_limit=30, question_length_limit=10, skip_impossible_questions=True
        )
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))
        assert len(instances[0].fields["question"].tokens) == 10
        assert len(instances[0].fields["passage"].tokens) == 30
        # invalid examples where all the answers exceed the passage length should be skipped.
        assert len(instances) == 3

        # Length limit still works if we do not skip the invalid examples
        reader = SquadReader(
            passage_length_limit=30, question_length_limit=10, skip_impossible_questions=False
        )
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))
        assert len(instances[0].fields["question"].tokens) == 10
        assert len(instances[0].fields["passage"].tokens) == 30
        # invalid examples should not be skipped.
        assert len(instances) == 5

        # Make sure the answer texts does not change, so that the evaluation will not be affected
        reader_unlimited = SquadReader(
            passage_length_limit=30, question_length_limit=10, skip_impossible_questions=False
        )
        instances_unlimited = ensure_list(
            reader_unlimited.read(FIXTURES_ROOT / "rc" / "squad.json")
        )
        for instance_x, instance_y in zip(instances, instances_unlimited):
            print(instance_x.fields["metadata"]["answer_texts"])
            assert set(instance_x.fields["metadata"]["answer_texts"]) == set(
                instance_y.fields["metadata"]["answer_texts"]
            )


class TestSquad2Reader:
    def test_read_from_file(self):
        reader = SquadReader.squad2()
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad2.json"))
        assert len(instances) == 6

        assert [t.text for t in instances[0].fields["question"].tokens[:3]] == ["This", "is", "an"]
        assert [t.text for t in instances[0].fields["passage"].tokens[:3]] == [
            "Architecturally",
            ",",
            "the",
        ]
        assert [t.text for t in instances[0].fields["passage"].tokens[-4:]] == [
            "of",
            "Mary",
            ".",
            SQUAD2_NO_ANSWER_TOKEN,
        ]
        assert instances[0].fields["span_start"].sequence_index == 142
        assert instances[0].fields["span_end"].sequence_index == 142

        assert [t.text for t in instances[1].fields["question"].tokens[:3]] == ["To", "whom", "did"]
        assert [t.text for t in instances[1].fields["passage"].tokens[:3]] == [
            "Architecturally",
            ",",
            "the",
        ]
        assert [t.text for t in instances[1].fields["passage"].tokens[-4:]] == [
            "of",
            "Mary",
            ".",
            SQUAD2_NO_ANSWER_TOKEN,
        ]
        assert instances[1].fields["span_start"].sequence_index == 102
        assert instances[1].fields["span_end"].sequence_index == 104

        assert [t.text for t in instances[2].fields["question"].tokens[:3]] == [
            "What",
            "sits",
            "on",
        ]
        assert [t.text for t in instances[2].fields["passage"].tokens[:3]] == [
            "Architecturally",
            ",",
            "the",
        ]
        assert [t.text for t in instances[2].fields["passage"].tokens[-4:]] == [
            "of",
            "Mary",
            ".",
            SQUAD2_NO_ANSWER_TOKEN,
        ]
        assert instances[2].fields["span_start"].sequence_index == 17
        assert instances[2].fields["span_end"].sequence_index == 23

        # We're checking this case because I changed the answer text to only have a partial
        # annotation for the last token, which happens occasionally in the training data.  We're
        # making sure we get a reasonable output in that case here.
        assert [t.text for t in instances[4].fields["question"].tokens[:3]] == [
            "Which",
            "individual",
            "worked",
        ]
        assert [t.text for t in instances[4].fields["passage"].tokens[:3]] == ["In", "1882", ","]
        assert [t.text for t in instances[4].fields["passage"].tokens[-4:]] == [
            "Nuclear",
            "Astrophysics",
            ".",
            SQUAD2_NO_ANSWER_TOKEN,
        ]
        span_start = instances[4].fields["span_start"].sequence_index
        span_end = instances[4].fields["span_end"].sequence_index
        answer_tokens = instances[4].fields["passage"].tokens[span_start : (span_end + 1)]
        expected_answer_tokens = ["Father", "Julius", "Nieuwland"]
        assert [t.text for t in answer_tokens] == expected_answer_tokens

    def test_can_build_from_params(self):
        reader = DatasetReader.from_params(Params({"type": "squad2"}))

        assert reader._tokenizer.__class__.__name__ == "SpacyTokenizer"  # type: ignore
        assert reader._token_indexers["tokens"].__class__.__name__ == "SingleIdTokenIndexer"  # type: ignore

    def test_length_limit_works(self):
        # We're making sure the length of the text is correct if length limit is provided.
        reader = SquadReader.squad2(
            passage_length_limit=30, question_length_limit=10, skip_impossible_questions=True
        )
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad2.json"))
        assert len(instances[0].fields["question"].tokens) == 6
        assert len(instances[0].fields["passage"].tokens) == 30
        # invalid examples where all the answers exceed the passage length should be skipped.
        assert len(instances) == 4

        # Length limit still works if we do not skip the invalid examples
        reader = SquadReader.squad2(
            passage_length_limit=30, question_length_limit=10, skip_impossible_questions=False
        )
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad2.json"))
        assert len(instances[0].fields["question"].tokens) == 6
        assert len(instances[0].fields["passage"].tokens) == 30
        # invalid examples should not be skipped.
        assert len(instances) == 6

        # Make sure the answer texts does not change, so that the evaluation will not be affected
        reader_unlimited = SquadReader.squad2(
            passage_length_limit=30, question_length_limit=10, skip_impossible_questions=False
        )
        instances_unlimited = ensure_list(
            reader_unlimited.read(FIXTURES_ROOT / "rc" / "squad2.json")
        )
        for instance_x, instance_y in zip(instances, instances_unlimited):
            print(instance_x.fields["metadata"]["answer_texts"])
            assert set(instance_x.fields["metadata"]["answer_texts"]) == set(
                instance_y.fields["metadata"]["answer_texts"]
            )
