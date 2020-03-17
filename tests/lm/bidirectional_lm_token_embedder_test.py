from tests import FIXTURES_ROOT
from tests.lm.language_model_token_embedder_test import TestLanguageModelTokenEmbedder


class TestBidirectionalLanguageModelTokenEmbedder(TestLanguageModelTokenEmbedder):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            FIXTURES_ROOT
            / "lm"
            / "language_model"
            / "bidirectional_lm_characters_token_embedder.jsonnet",
            FIXTURES_ROOT / "lm" / "conll2003.txt",
        )


class TestBidirectionalLanguageModelTokenEmbedderWithoutBosEos(TestLanguageModelTokenEmbedder):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            FIXTURES_ROOT
            / "lm"
            / "language_model"
            / "bidirectional_lm_characters_token_embedder_without_bos_eos.jsonnet",
            FIXTURES_ROOT / "lm" / "conll2003.txt",
        )
