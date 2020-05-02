from allennlp.common.testing.model_test_case import ModelTestCase
from tests import FIXTURES_ROOT

import allennlp_models.syntax


class GraphParserTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            FIXTURES_ROOT / "syntax" / "semantic_dependencies" / "experiment.json",
            FIXTURES_ROOT / "syntax" / "semantic_dependencies" / "dm.sdp",
        )

    def test_graph_parser_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_model_can_decode(self):
        self.model.eval()
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.make_output_human_readable(output_dict)

        assert set(decode_output_dict.keys()) == {
            "arc_loss",
            "tag_loss",
            "loss",
            "arcs",
            "arc_tags",
            "arc_tag_probs",
            "arc_probs",
            "tokens",
            "mask",
        }
