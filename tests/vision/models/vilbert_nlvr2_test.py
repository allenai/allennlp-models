from torch.testing import assert_allclose
from transformers import AutoModel

from allennlp.common.testing import ModelTestCase
from allennlp.data import Vocabulary
from allennlp.common.testing import assert_equal_parameters

from allennlp_models import vision  # noqa: F401

from tests import FIXTURES_ROOT


class TestNlvr2Vilbert(ModelTestCase):
    def test_model_can_train_save_and_load_small_model(self):
        param_file = FIXTURES_ROOT / "vision" / "nlvr2" / "experiment.jsonnet"
        self.ensure_model_can_train_save_and_load(
            param_file, gradients_to_ignore={"classifier.weight", "classifier.bias"}
        )

    def test_model_can_train_save_and_load_with_cache(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix=self.__class__.__name__) as d:
            overrides = {"dataset_reader.feature_cache_dir": str(d)}
            import json

            overrides = json.dumps(overrides)
            param_file = FIXTURES_ROOT / "vision" / "nlvr2" / "experiment.jsonnet"
            self.ensure_model_can_train_save_and_load(
                param_file,
                overrides=overrides,
                gradients_to_ignore={"classifier.weight", "classifier.bias"},
            )

    def test_model_can_train_save_and_load_from_huggingface(self):
        param_file = FIXTURES_ROOT / "vision" / "nlvr2" / "experiment_from_huggingface.jsonnet"
        self.ensure_model_can_train_save_and_load(
            param_file, gradients_to_ignore={"classifier.weight", "classifier.bias"}
        )

    def test_model_loads_weights_correctly(self):
        from allennlp_models.vision.models.nlvr2 import Nlvr2Model

        vocab = Vocabulary()
        model_name = "epwalsh/bert-xsmall-dummy"
        model = Nlvr2Model.from_huggingface_model_name(
            vocab=vocab,
            model_name=model_name,
            image_feature_dim=2048,
            image_num_hidden_layers=1,
            image_hidden_size=3,
            image_num_attention_heads=1,
            combined_num_attention_heads=1,
            combined_hidden_size=5,
            pooled_output_dim=7,
            image_intermediate_size=11,
            image_attention_dropout=0.0,
            image_hidden_dropout=0.0,
            image_biattention_id=[0, 1],
            text_biattention_id=[0, 1],
            text_fixed_layer=0,
            image_fixed_layer=0,
        )

        transformer = AutoModel.from_pretrained(model_name)

        # compare embedding parameters
        assert_allclose(
            transformer.embeddings.word_embeddings.weight.data,
            model.backbone.text_embeddings.embeddings.word_embeddings.weight.data,
        )

        # compare encoder parameters
        assert_allclose(
            transformer.encoder.layer[0].intermediate.dense.weight.data,
            model.backbone.encoder.layers1[0].intermediate.dense.weight.data,
        )
