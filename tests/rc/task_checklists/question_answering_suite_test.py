import pytest
from allennlp.sanity_checks.task_checklists.question_answering_suite import QuestionAnsweringSuite
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from allennlp_models.rc.predictors import *  # noqa: F403
from tests import FIXTURES_ROOT


class TestQuestionAnsweringSuite(AllenNlpTestCase):
    @pytest.mark.parametrize(
        "model",
        [
            "bidaf",
        ],
    )
    def test_run(self, model: str):
        archive = load_archive(FIXTURES_ROOT / "rc" / model / "serialization" / "model.tar.gz")
        predictor = Predictor.from_archive(archive)

        data = [
            ("Alice is taller than Bob.", "Who is taller?"),
            ("Children were playing in the park.", "Was the park empty?"),
        ]
        suite = QuestionAnsweringSuite(context_key="passage", add_default_tests=True, data=data)
        suite.run(predictor, max_examples=10)
