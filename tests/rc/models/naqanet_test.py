from allennlp.common.testing import ModelTestCase

from tests import FIXTURES_ROOT

from allennlp_models.rc import NumericallyAugmentedQaNet


class NumericallyAugmentedQaNetTest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "rc" / "naqanet" / "experiment.json",
            FIXTURES_ROOT / "rc" / "drop.json",
        )

    def test_model_can_train_save_and_load(self):
        import torch

        torch.autograd.set_detect_anomaly(True)
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            # Due to numerical instability, these scalar tensors might sometimes
            # have zero gradients.
            gradients_to_ignore={
                "_passage_span_end_predictor._linear_layers.1.bias",
                "_question_span_end_predictor._linear_layers.1.bias",
            },
        )
