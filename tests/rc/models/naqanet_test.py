from flaky import flaky

from allennlp.common.testing import ModelTestCase

from tests import FIXTURES_ROOT

from allennlp_models.rc import NumericallyAugmentedQaNet


class NumericallyAugmentedQaNetTest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "rc" / "naqanet" / "experiment.json", FIXTURES_ROOT / "rc" / "drop.json"
        )

    @flaky(max_runs=3, min_passes=1)
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            gradients_to_ignore={
                "_passage_span_end_predictor._linear_layers.1.bias",
                "_question_span_end_predictor._linear_layers.1.bias",
            },
        )
