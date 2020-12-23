from allennlp.common.testing import ModelTestCase

from tests import FIXTURES_ROOT
from allennlp_models import lm  # noqa: F401


class TestNextTokenLanguageModel(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "lm" / "next_token_lm" / "experiment.json",
            FIXTURES_ROOT / "lm" / "language_model" / "sentences.txt",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


class TestNextTokenTransformerLm(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "lm" / "next_token_lm" / "experiment_transformer.json",
            FIXTURES_ROOT / "lm" / "language_model" / "sentences.txt",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            tolerance=1e-3,
            gradients_to_ignore={
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.weight",
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.bias",
            },
        )
