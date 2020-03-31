from allennlp.common.testing import ModelTestCase
from allennlp.data import Batch

from tests import FIXTURES_ROOT

import allennlp_models.rc.dialog_qa


class DialogQATest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            FIXTURES_ROOT / "rc" / "dialog_qa" / "experiment.json",
            FIXTURES_ROOT / "rc" / "dialog_qa" / "quac_sample.json",
        )
        self.batch = Batch(self.instances)
        self.batch.index_instances(self.vocab)

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert "best_span_str" in output_dict and "loss" in output_dict
        assert "followup" in output_dict and "yesno" in output_dict

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
