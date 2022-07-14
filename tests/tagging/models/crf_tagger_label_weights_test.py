from flaky import flaky
import pytest

from allennlp.commands.train import train_model_from_file
from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError

from tests import FIXTURES_ROOT


class CrfTaggerLabelWeightsTest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "tagging" / "crf_tagger" / "experiment.json",
            FIXTURES_ROOT / "tagging" / "conll2003.txt",
        )

    def test_label_weights_effectiveness(self):
        training_tensors = self.dataset.as_tensor_dict()
        save_dir = self.TEST_DIR / "save_and_load_test"

        # original CRF
        output_dict_original = self.model(**training_tensors)

        # weighted CRF
        model_weighted = train_model_from_file(
            self.param_file,
            save_dir,
            overrides={"model.label_weights": {"I-ORG": 10.0}},
            force=True,
            return_model=True,
        )
        output_dict_weighted = model_weighted(**training_tensors)

        # assert that logits are substantially different
        assert (
            output_dict_weighted["logits"].isclose(output_dict_original["logits"]).sum()
            < output_dict_original["logits"].numel() / 2
        )

    def test_label_weights_effectiveness_emission_transition(self):
        training_tensors = self.dataset.as_tensor_dict()
        save_dir = self.TEST_DIR / "save_and_load_test"

        # original CRF
        output_dict_original = self.model(**training_tensors)

        # weighted CRF
        model_weighted = train_model_from_file(
            self.param_file,
            save_dir,
            overrides={
                "model.label_weights": {"I-ORG": 10.0},
                "model.weight_strategy": "emission_transition",
            },
            force=True,
            return_model=True,
        )
        output_dict_weighted = model_weighted(**training_tensors)

        # assert that logits are substantially different
        assert (
            output_dict_weighted["logits"].isclose(output_dict_original["logits"]).sum()
            < output_dict_original["logits"].numel() / 2
        )

    def test_label_weights_effectiveness_lannoy(self):
        training_tensors = self.dataset.as_tensor_dict()
        save_dir = self.TEST_DIR / "save_and_load_test"

        # original CRF
        output_dict_original = self.model(**training_tensors)

        # weighted CRF
        model_weighted = train_model_from_file(
            self.param_file,
            save_dir,
            overrides={
                "model.label_weights": {"I-ORG": 10.0},
                "model.weight_strategy": "lannoy",
            },
            force=True,
            return_model=True,
        )
        output_dict_weighted = model_weighted(**training_tensors)

        # assert that logits are substantially different
        assert (
            output_dict_weighted["logits"].isclose(output_dict_original["logits"]).sum()
            < output_dict_original["logits"].numel() / 2
        )

    def test_config_error_invalid_label(self):
        save_dir = self.TEST_DIR / "save_and_load_test"
        with pytest.raises(ConfigurationError):
            train_model_from_file(
                self.param_file,
                save_dir,
                overrides={"model.label_weights": {"BLA": 10.0}},
                force=True,
                return_model=True,
            )

    def test_config_error_strategy_without_weights(self):
        save_dir = self.TEST_DIR / "save_and_load_test"
        with pytest.raises(ConfigurationError):
            train_model_from_file(
                self.param_file,
                save_dir,
                overrides={"model.weight_strategy": "emission"},
                force=True,
                return_model=True,
            )

    def test_config_error_invalid_strategy(self):
        save_dir = self.TEST_DIR / "save_and_load_test"
        with pytest.raises(ConfigurationError):
            train_model_from_file(
                self.param_file,
                save_dir,
                overrides={
                    "model.label_weights": {"I-ORG": 10.0},
                    "model.weight_strategy": "invalid",
                },
                force=True,
                return_model=True,
            )
