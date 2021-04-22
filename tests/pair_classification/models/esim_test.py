import numpy
from numpy.testing import assert_almost_equal

from allennlp.common.testing import ModelTestCase

from tests import FIXTURES_ROOT


class TestESIM(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "pair_classification" / "esim" / "experiment.json",
            FIXTURES_ROOT / "pair_classification" / "snli.jsonl",
        )

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert_almost_equal(numpy.sum(output_dict["label_probs"][0].data.numpy(), -1), 1, decimal=6)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_decode_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.make_output_human_readable(output_dict)
        assert "label" in decode_output_dict
