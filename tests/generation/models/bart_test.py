import pytest
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.models import Model
from allennlp_models import generation  # noqa: F401
from tests import FIXTURES_ROOT


class BartTest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "generation" / "bart" / "experiment.jsonnet",
            FIXTURES_ROOT / "generation" / "bart" / "data" / "url_lists" / "all_train.txt",
        )

    def test_backwards_compatibility_with_beam_search_args(self):
        # These values are arbitrary but should be different than the config.
        beam_size, max_decoding_steps = 100, 1000
        params = Params.from_file(self.param_file)
        params["model"]["beam_size"] = beam_size
        params["model"]["max_decoding_steps"] = max_decoding_steps
        # The test harness is set up to treat DeprecationWarning's like errors, so this needs to
        # be called within the pytest context manager.
        with pytest.raises(DeprecationWarning):
            model = Model.from_params(vocab=self.vocab, params=params.get("model"))
            assert model._beam_search.beam_size == beam_size
            assert model._beam_search.max_steps == max_decoding_steps

    def test_model_can_train_save_load_predict(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)
