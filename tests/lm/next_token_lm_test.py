from allennlp.common.testing import ModelTestCase

from allennlp_models.lm.language_model_heads import LinearLanguageModelHead
from tests import FIXTURES_ROOT


class TestNextTokenLanguageModel(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            FIXTURES_ROOT / "lm" / "next_token_lm" / "experiment.json",
            FIXTURES_ROOT / "lm" / "language_model" / "sentences.txt",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
