from allennlp.common.testing import ModelTestCase
from allennlp.data.batch import Batch

from tests import FIXTURES_ROOT


class TestLanguageModelTokenEmbedder(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "lm" / "language_model" / "characters_token_embedder.json",
            FIXTURES_ROOT / "lm" / "conll2003.txt",
        )

    def test_tagger_with_language_model_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_language_model_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict["tags"]
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {"O", "I-ORG", "I-PER", "I-LOC"}


class TestLanguageModelTokenEmbedderWithoutBosEos(TestLanguageModelTokenEmbedder):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT
            / "lm"
            / "language_model"
            / "characters_token_embedder_without_bos_eos.jsonnet",
            FIXTURES_ROOT / "lm" / "conll2003.txt",
        )
