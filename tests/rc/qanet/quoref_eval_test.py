import os

from allennlp.common.testing import AllenNlpTestCase

from allennlp_models.rc.qanet import quoref_eval
from tests import FIXTURES_ROOT


class TestQuorefEval(AllenNlpTestCase):
    """
    The actual evaluation logic in Quoref's evaluation script is from DROP's script, and the
    only additional thing that Quoref's script does is handling the data properly. So this class only tests the
    data handling aspects. The tests we have for DROP are fairly comprehensive.
    """

    def test_quoref_eval_with_original_data_format(self):
        predictions_file = FIXTURES_ROOT / "rc" / "quoref_sample_predictions.json"
        gold_file = FIXTURES_ROOT / "rc" / "quoref_sample.json"
        metrics = quoref_eval.evaluate_prediction_file(predictions_file, gold_file)
        assert metrics == (0.5, 0.625)

    def test_quoref_eval_with_simple_format(self):
        predictions_file = FIXTURES_ROOT / "rc" / "quoref_sample_predictions.json"
        gold_file = FIXTURES_ROOT / "rc" / "quoref_sample_predictions.json"
        metrics = quoref_eval.evaluate_prediction_file(predictions_file, gold_file)
        assert metrics == (1.0, 1.0)

    def test_quoref_eval_script(self):
        predictions_file = FIXTURES_ROOT / "rc" / "quoref_sample_predictions.json"
        gold_file = FIXTURES_ROOT / "rc" / "quoref_sample.json"
        result = os.system(
            f"python -m allennlp_models.rc.qanet.quoref_eval --gold_path {gold_file} "
            f"--prediction_path {predictions_file} --output_path /tmp/output.json"
        )
        assert result == 0
