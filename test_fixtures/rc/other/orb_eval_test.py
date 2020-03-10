# pylint: disable=no-self-use,invalid-name

from allennlp_models.rc.common.squad_eval import normalize_answer as _normalize_answer_squad
from allennlp_models.rc.other.orb_utils import get_metric_squad, get_metric_drop
from allennlp_models.rc.common.squad2_eval import get_metric_score as get_metric_squad2
from allennlp_models.rc.other.narrativeqa_eval import get_metric_score as get_metric_narrativeqa
from tests import FIXTURES_ROOT
import os


class TestSQUAD1:
    def test_spaces_are_ignored(self):
        assert _normalize_answer_squad("abcd") == _normalize_answer_squad("abcd  ")
        assert _normalize_answer_squad("abcd") == _normalize_answer_squad("  abcd  ")
        assert _normalize_answer_squad("  ABCD") == _normalize_answer_squad("ABCD")

    def test_punctations_are_ignored(self):
        assert _normalize_answer_squad("T.J Howard") == _normalize_answer_squad("tj howard")
        assert _normalize_answer_squad("7802") == _normalize_answer_squad("78.02")

    def test_articles_are_ignored(self):
        assert get_metric_squad("td", ["the td"]) == (1.0, 1.0)
        assert get_metric_squad("the a NOT an ARTICLE the an a", ["NOT ARTICLE"]) == (1.0, 1.0)

    def test_casing_is_ignored(self):
        assert get_metric_squad("This was a triumph", ["tHIS Was A TRIUMPH"]) == (1.0, 1.0)


class TestDROP:
    def test_articles_are_ignored(self):
        assert get_metric_drop("td", ["the td"]) == (1.0, 1.0)
        assert get_metric_drop("the a NOT an ARTICLE the an a", ["NOT ARTICLE"]) == (1.0, 1.0)

    def test_casing_is_ignored(self):
        assert get_metric_drop("This was a triumph", ["tHIS Was A TRIUMPH"]) == (1.0, 1.0)

    def test_long_answers(self):
        assert (
            get_metric_drop(
                "David Thomas",
                [
                    "Thomas David Arquette Thomas David Arquette Thomas \
                    David Arquette Thomas David Arquette"
                ],
            )
            == (0.0, 0.8)
        )

    def test_span_order_is_ignored(self):
        assert get_metric_drop(["athlete", "unprofessional"], [["unprofessional", "athlete"]]) == (
            1.0,
            1.0,
        )
        assert get_metric_drop(
            ["algebra", "arithmetic"], [["arithmetic", "algebra", "geometry"]]
        ) == (0.0, 0.67)

    def test_word_order_is_not_ignored(self):
        assert get_metric_drop(["athlete unprofessional"], [["unprofessional athlete"]]) == (
            0.0,
            1.0,
        )

    def test_bag_alignment_is_optimal(self):
        assert get_metric_drop(
            ["Thomas Jefferson", "Thomas Davidson Arquette"], [["David Thomas", "Thomas Jefferson"]]
        ) == (0.0, 0.7)
        assert get_metric_drop(
            ["Thomas David Arquette"], [["David Thomas", "Thomas Jefferson"]]
        ) == (0.0, 0.4)

    def test_multiple_gold_spans(self):
        assert get_metric_drop(
            ["Thomas David Arquette"],
            [["David Thomas"], ["Thomas Jefferson"], ["David Thomas"], ["Thomas David"]],
        ) == (0.0, 0.8)

    def test_long_gold_spans(self):
        assert get_metric_drop(
            ["Thomas David Arquette"], [["David Thomas was eating an apple and fell to the ground"]]
        ) == (0.0, 0.33)


class TestNarrativeQA:
    def test_ngrams(self):
        assert get_metric_narrativeqa(
            "David Thomas was eating an apple",
            ["David Thomas was eating an apple and fell to the ground"],
        ) == (0.43, 0.43, 0.57, 0.75, 1.0, 0.6)
        assert get_metric_narrativeqa(
            "David Thomas was eating an apple and fell to the ground",
            ["David Thomas was eating an apple", "he fell to the ground"],
        ) == (0.55, 0.38, 0.92, 0.75, 0.6, 1.0)
        assert get_metric_narrativeqa(
            "David Thomas was eating an apple and fell to the ground",
            ["David Thomas was eating an apple and fell to the ground"],
        ) == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


class TestQuoREF:
    def test_articles_are_ignored(self):
        assert get_metric_drop("td", ["the td"]) == (1.0, 1.0)
        assert get_metric_drop("the a NOT an ARTICLE the an a", ["NOT ARTICLE"]) == (1.0, 1.0)

    def test_casing_is_ignored(self):
        assert get_metric_drop("This was a triumph", ["tHIS Was A TRIUMPH"]) == (1.0, 1.0)

    def test_long_answers(self):
        assert (
            get_metric_drop(
                "David Thomas",
                [
                    "Thomas David Arquette Thomas David Arquette Thomas \
                    David Arquette Thomas David Arquette"
                ],
            )
            == (0.0, 0.8)
        )

    def test_span_order_is_ignored(self):
        assert get_metric_drop(["athlete", "unprofessional"], [["unprofessional", "athlete"]]) == (
            1.0,
            1.0,
        )
        assert get_metric_drop(
            ["algebra", "arithmetic"], [["arithmetic", "algebra", "geometry"]]
        ) == (0.0, 0.67)

    def test_word_order_is_not_ignored(self):
        assert get_metric_drop(["athlete unprofessional"], [["unprofessional athlete"]]) == (
            0.0,
            1.0,
        )

    def test_bag_alignment_is_optimal(self):
        assert get_metric_drop(
            ["Thomas Jefferson", "Thomas Davidson Arquette"], [["David Thomas", "Thomas Jefferson"]]
        ) == (0.0, 0.7)
        assert get_metric_drop(
            ["Thomas David Arquette"], [["David Thomas", "Thomas Jefferson"]]
        ) == (0.0, 0.4)

    def test_multiple_gold_spans(self):
        assert get_metric_drop(
            ["Thomas David Arquette"],
            [["David Thomas"], ["Thomas Jefferson"], ["David Thomas"], ["Thomas David"]],
        ) == (0.0, 0.8)

    def test_long_gold_spans(self):
        assert get_metric_drop(
            ["Thomas David Arquette"], [["David Thomas was eating an apple and fell to the ground"]]
        ) == (0.0, 0.33)


class TestSQUAD2:
    def test_impossible_answer(self):
        assert get_metric_squad2("", ["news"]) == (0.0, 0.0)
        assert get_metric_squad2("news", [""]) == (0.0, 0.0)
        assert get_metric_squad2("", [""]) == (1.0, 1.0)

    def test_functional_case(self):
        assert get_metric_squad2("This was a triumph", ["a triumph"]) == (0.0, 0.5)


class TestIntegration:
    def test_sample_results(self):
        gold_file = FIXTURES_ROOT / "rc" / "orb_sample_input.jsonl"
        predictions_file = FIXTURES_ROOT / "rc" / "orb_sample_predictions.json"
        result = os.system(
            f"python -m allennlp_rc.eval.orb_eval --dataset_file {gold_file} "
            f"--prediction_file {predictions_file} --metrics_output_file /tmp/output.json"
        )
        assert result == 0
