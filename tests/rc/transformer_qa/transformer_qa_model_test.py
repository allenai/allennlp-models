from flaky import flaky
import numpy
from numpy.testing import assert_almost_equal
from allennlp.common.testing import ModelTestCase
from allennlp.data import Batch
from tests import FIXTURES_ROOT

import allennlp_models.rc.transformer_qa  # noqa F401: Needed to register the registrables.


class TransformerQaTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            FIXTURES_ROOT / "rc" / "transformer_qa" / "experiment.jsonnet",
            FIXTURES_ROOT / "rc" / "squad.json",
        )

    def test_model_can_train_save_and_load(self):
        # Huggingface transformer models come with pooler weights, but this model doesn't use the pooler.
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            gradients_to_ignore={
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.weight",
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.bias",
            },
        )

    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)

        metrics = self.model.get_metrics(reset=True)
        # We've set up the data such that there's a fake answer that consists of the whole
        # paragraph.  _Any_ valid prediction for that question should produce an F1 of greater than
        # zero, while if we somehow haven't been able to load the evaluation data, or there was an
        # error with using the evaluation script, this will fail.  This makes sure that we've
        # loaded the evaluation data correctly and have hooked things up to the official evaluation
        # script.
        assert metrics["per_instance_f1"] > 0

        span_start_probs = output_dict["span_start_probs"][0].data.numpy()
        span_end_probs = output_dict["span_start_probs"][0].data.numpy()
        assert_almost_equal(numpy.sum(span_start_probs, -1), 1, decimal=6)
        assert_almost_equal(numpy.sum(span_end_probs, -1), 1, decimal=6)
        span_start, span_end = tuple(output_dict["best_span"][0].data.numpy())
        assert span_start >= 0
        assert span_start <= span_end
        assert span_end < self.instances[0].fields["question_with_context"].sequence_length()
        assert isinstance(output_dict["best_span_str"][0], str)
