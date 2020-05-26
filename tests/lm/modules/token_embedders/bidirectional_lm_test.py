from tests import FIXTURES_ROOT
from tests.lm.modules.token_embedders.language_model_test import TestLanguageModelTokenEmbedder


class TestBidirectionalLanguageModelTokenEmbedder(TestLanguageModelTokenEmbedder):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT
            / "lm"
            / "language_model"
            / "bidirectional_lm_characters_token_embedder.jsonnet",
            FIXTURES_ROOT / "lm" / "conll2003.txt",
        )


class TestBidirectionalLanguageModelTokenEmbedderWithoutBosEos(TestLanguageModelTokenEmbedder):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT
            / "lm"
            / "language_model"
            / "bidirectional_lm_characters_token_embedder_without_bos_eos.jsonnet",
            FIXTURES_ROOT / "lm" / "conll2003.txt",
        )
