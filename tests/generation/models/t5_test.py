from allennlp.common.testing import ModelTestCase

from tests import FIXTURES_ROOT

from allennlp_models import generation  # noqa: F401


class T5Test(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "generation" / "t5" / "experiment.jsonnet",
            FIXTURES_ROOT / "generation" / "bart" / "data" / "url_lists" / "all_train.txt",
        )

    def test_model_can_train_save_load_predict(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)
