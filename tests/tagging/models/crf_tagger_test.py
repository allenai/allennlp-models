from flaky import flaky
import pytest

from allennlp.commands.train import train_model_from_file
from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model

from tests import FIXTURES_ROOT


class CrfTaggerTest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "tagging" / "crf_tagger" / "experiment.json",
            FIXTURES_ROOT / "tagging" / "conll2003.txt",
        )

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_simple_tagger_can_train_save_and_load_ccgbank(self):
        self.ensure_model_can_train_save_and_load(
            FIXTURES_ROOT / "tagging" / "crf_tagger" / "experiment_ccgbank.json"
        )

    def test_simple_tagger_can_train_save_and_conll2000(self):
        self.ensure_model_can_train_save_and_load(
            FIXTURES_ROOT / "tagging" / "crf_tagger" / "experiment_conll2000.json"
        )

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict["tags"]
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {"O", "I-ORG", "I-PER", "I-LOC"}

    def test_low_loss_for_pretrained_transformers(self):
        self.set_up_model(
            FIXTURES_ROOT / "tagging" / "crf_tagger" / "experiment_albert.json",
            FIXTURES_ROOT / "tagging" / "conll2003.txt",
        )
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)

        assert output_dict["loss"] < 50

    def test_forward_pass_top_k(self):
        training_tensors = self.dataset.as_tensor_dict()
        self.model.top_k = 5
        output_dict = self.model.make_output_human_readable(self.model(**training_tensors))
        top_k_tags = [[x["tags"] for x in item_topk] for item_topk in output_dict["top_k_tags"]]
        first_choices = [x[0] for x in top_k_tags]
        assert first_choices == output_dict["tags"]
        lengths = [len(x) for x in top_k_tags]
        assert set(lengths) == {5}
        tags_used = set(
            tag for item_top_k in top_k_tags for tag_seq in item_top_k for tag in tag_seq
        )
        assert all(tag in {"O", "I-ORG", "I-PER", "I-LOC"} for tag in tags_used)

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_token_based_verbose_metrics(self):
        training_tensors = self.dataset.as_tensor_dict()
        save_dir = self.TEST_DIR / "save_and_load_test"

        model = train_model_from_file(
            self.param_file,
            save_dir,
            overrides={
                "model.calculate_span_f1": False,
                "model.verbose_metrics": True,
            },
            force=True,
            return_model=True,
        )
        model(**training_tensors)
        metrics = model.get_metrics()

        # assert that metrics contain all verbose keys
        for tag in ["O", "I-PER", "I-ORG", "I-LOC", "micro", "macro", "weighted"]:
            for m in ["precision", "recall", "fscore"]:
                assert f"{tag}-{m}" in metrics
