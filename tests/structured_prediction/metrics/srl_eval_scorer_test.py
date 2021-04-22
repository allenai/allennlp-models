import pytest
from numpy.testing import assert_allclose

from allennlp.common.testing import (
    AllenNlpTestCase,
    global_distributed_metric,
    run_distributed_test,
)

from allennlp_models.structured_prediction.metrics.srl_eval_scorer import SrlEvalScorer


class SrlEvalScorerTest(AllenNlpTestCase):
    def test_srl_eval_correctly_scores_identical_tags(self):
        batch_verb_indices = [3, 8, 2, 0]
        batch_sentences = [
            [
                "Mali",
                "government",
                "officials",
                "say",
                "the",
                "woman",
                "'s",
                "confession",
                "was",
                "forced",
                ".",
            ],
            [
                "Mali",
                "government",
                "officials",
                "say",
                "the",
                "woman",
                "'s",
                "confession",
                "was",
                "forced",
                ".",
            ],
            [
                "The",
                "prosecution",
                "rested",
                "its",
                "case",
                "last",
                "month",
                "after",
                "four",
                "months",
                "of",
                "hearings",
                ".",
            ],
            ["Come", "in", "and", "buy", "."],
        ]
        batch_bio_predicted_tags = [
            [
                "B-ARG0",
                "I-ARG0",
                "I-ARG0",
                "B-V",
                "B-ARG1",
                "I-ARG1",
                "I-ARG1",
                "I-ARG1",
                "I-ARG1",
                "I-ARG1",
                "O",
            ],
            ["O", "O", "O", "O", "B-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "B-V", "B-ARG2", "O"],
            [
                "B-ARG0",
                "I-ARG0",
                "B-V",
                "B-ARG1",
                "I-ARG1",
                "B-ARGM-TMP",
                "I-ARGM-TMP",
                "B-ARGM-TMP",
                "I-ARGM-TMP",
                "I-ARGM-TMP",
                "I-ARGM-TMP",
                "I-ARGM-TMP",
                "O",
            ],
            ["B-V", "B-AM-DIR", "O", "O", "O"],
        ]
        from allennlp_models.structured_prediction.models.srl import (
            convert_bio_tags_to_conll_format,
        )

        batch_conll_predicted_tags = [
            convert_bio_tags_to_conll_format(tags) for tags in batch_bio_predicted_tags
        ]
        batch_bio_gold_tags = [
            [
                "B-ARG0",
                "I-ARG0",
                "I-ARG0",
                "B-V",
                "B-ARG1",
                "I-ARG1",
                "I-ARG1",
                "I-ARG1",
                "I-ARG1",
                "I-ARG1",
                "O",
            ],
            ["O", "O", "O", "O", "B-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "B-V", "B-ARG2", "O"],
            [
                "B-ARG0",
                "I-ARG0",
                "B-V",
                "B-ARG1",
                "I-ARG1",
                "B-ARGM-TMP",
                "I-ARGM-TMP",
                "B-ARGM-TMP",
                "I-ARGM-TMP",
                "I-ARGM-TMP",
                "I-ARGM-TMP",
                "I-ARGM-TMP",
                "O",
            ],
            ["B-V", "B-AM-DIR", "O", "O", "O"],
        ]
        batch_conll_gold_tags = [
            convert_bio_tags_to_conll_format(tags) for tags in batch_bio_gold_tags
        ]

        srl_scorer = SrlEvalScorer(ignore_classes=["V"])
        srl_scorer(
            batch_verb_indices, batch_sentences, batch_conll_predicted_tags, batch_conll_gold_tags
        )
        metrics = srl_scorer.get_metric()
        assert len(metrics) == 18
        assert_allclose(metrics["precision-ARG0"], 1.0)
        assert_allclose(metrics["recall-ARG0"], 1.0)
        assert_allclose(metrics["f1-measure-ARG0"], 1.0)
        assert_allclose(metrics["precision-ARG1"], 1.0)
        assert_allclose(metrics["recall-ARG1"], 1.0)
        assert_allclose(metrics["f1-measure-ARG1"], 1.0)
        assert_allclose(metrics["precision-ARG2"], 1.0)
        assert_allclose(metrics["recall-ARG2"], 1.0)
        assert_allclose(metrics["f1-measure-ARG2"], 1.0)
        assert_allclose(metrics["precision-ARGM-TMP"], 1.0)
        assert_allclose(metrics["recall-ARGM-TMP"], 1.0)
        assert_allclose(metrics["f1-measure-ARGM-TMP"], 1.0)
        assert_allclose(metrics["precision-AM-DIR"], 1.0)
        assert_allclose(metrics["recall-AM-DIR"], 1.0)
        assert_allclose(metrics["f1-measure-AM-DIR"], 1.0)
        assert_allclose(metrics["precision-overall"], 1.0)
        assert_allclose(metrics["recall-overall"], 1.0)
        assert_allclose(metrics["f1-measure-overall"], 1.0)

    def test_span_metrics_are_computed_correctly(self):
        from allennlp_models.structured_prediction.models.srl import (
            convert_bio_tags_to_conll_format,
        )

        batch_verb_indices = [2]
        batch_sentences = [["The", "cat", "loves", "hats", "."]]
        batch_bio_predicted_tags = [["B-ARG0", "B-ARG1", "B-V", "B-ARG1", "O"]]
        batch_conll_predicted_tags = [
            convert_bio_tags_to_conll_format(tags) for tags in batch_bio_predicted_tags
        ]
        batch_bio_gold_tags = [["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "O"]]
        batch_conll_gold_tags = [
            convert_bio_tags_to_conll_format(tags) for tags in batch_bio_gold_tags
        ]

        srl_scorer = SrlEvalScorer(ignore_classes=["V"])
        srl_scorer(
            batch_verb_indices, batch_sentences, batch_conll_predicted_tags, batch_conll_gold_tags
        )
        metrics = srl_scorer.get_metric()
        assert len(metrics) == 9
        assert_allclose(metrics["precision-ARG0"], 0.0)
        assert_allclose(metrics["recall-ARG0"], 0.0)
        assert_allclose(metrics["f1-measure-ARG0"], 0.0)
        assert_allclose(metrics["precision-ARG1"], 0.5)
        assert_allclose(metrics["recall-ARG1"], 1.0)
        assert_allclose(metrics["f1-measure-ARG1"], 2 / 3)
        assert_allclose(metrics["precision-overall"], 1 / 3)
        assert_allclose(metrics["recall-overall"], 1 / 2)
        assert_allclose(
            metrics["f1-measure-overall"], (2 * (1 / 3) * (1 / 2)) / ((1 / 3) + (1 / 2))
        )

    def test_distributed_setting_throws_an_error(self):
        from allennlp_models.structured_prediction.models.srl import (
            convert_bio_tags_to_conll_format,
        )

        batch_verb_indices = [2]
        batch_sentences = [["The", "cat", "loves", "hats", "."]]
        batch_bio_predicted_tags = [["B-ARG0", "B-ARG1", "B-V", "B-ARG1", "O"]]
        batch_conll_predicted_tags = [
            convert_bio_tags_to_conll_format(tags) for tags in batch_bio_predicted_tags
        ]
        batch_bio_gold_tags = [["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "O"]]
        batch_conll_gold_tags = [
            convert_bio_tags_to_conll_format(tags) for tags in batch_bio_gold_tags
        ]

        metric_kwargs = {
            "batch_verb_indices": [batch_verb_indices, batch_verb_indices],
            "batch_sentences": [batch_sentences, batch_sentences],
            "batch_conll_formatted_predicted_tags": [
                batch_conll_predicted_tags,
                batch_conll_predicted_tags,
            ],
            "batch_conll_formatted_gold_tags": [batch_conll_gold_tags, batch_conll_gold_tags],
        }

        desired_values = {}  # it does not matter, we expect the run to fail.

        with pytest.raises(Exception) as exc:
            run_distributed_test(
                [-1, -1],
                global_distributed_metric,
                SrlEvalScorer(ignore_classes=["V"]),
                metric_kwargs,
                desired_values,
                exact=True,
            )
            assert (
                "RuntimeError: Distributed aggregation for `SrlEvalScorer` is currently not supported."
                in str(exc.value)
            )
