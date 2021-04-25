import pytest
from allennlp.sanity_checks.task_checklists.textual_entailment_suite import TextualEntailmentSuite
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from allennlp_models.pair_classification.predictors import *  # noqa: F403
from tests import FIXTURES_ROOT


class TestTextualEntailmentSuite(AllenNlpTestCase):
    @pytest.mark.parametrize(
        "model",
        [
            "decomposable_attention",
            "esim",
        ],
    )
    def test_run(self, model: str):

        archive = load_archive(
            FIXTURES_ROOT / "pair_classification" / model / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive)

        data = [
            ("Alice and Bob are friends.", "Alice is Bob's friend."),
            ("The park had children playing", "The park was empty."),
        ]

        suite = TextualEntailmentSuite(probs_key="label_probs", add_default_tests=True, data=data)
        suite.run(predictor, max_examples=10)
