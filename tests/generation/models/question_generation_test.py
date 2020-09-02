from allennlp.common.testing import ModelTestCase

from tests import FIXTURES_ROOT


class QuestionGenerationModelTest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "generation" / "question_generation" / "experiment.jsonnet",
            FIXTURES_ROOT / "rc" / "squad.json",
        )

    def test_model_can_train_save_load_predict(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)
