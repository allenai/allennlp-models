import numpy
from numpy.testing import assert_almost_equal
from allennlp.commands.train import train_model_from_file
from allennlp.common.testing import ModelTestCase, AllenNlpTestCase, requires_gpu
from allennlp.data import Batch
from tests import FIXTURES_ROOT
import pytest
import torch

import allennlp_models.rc


class TransformerQaTest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "rc" / "transformer_qa" / "experiment.jsonnet",
            FIXTURES_ROOT / "rc" / "squad.json",
        )

    def test_model_can_train_save_and_load(self):
        # Huggingface transformer models come with pooler weights, but this model doesn't use the pooler.
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            gradients_to_ignore={
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.weight",
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.bias",
            },
        )

    def test_forward_pass_runs_correctly(self):
        self.model.training = False
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

        output_span_start_probs = output_dict["span_start_probs"][0].data.numpy()
        output_span_end_probs = output_dict["span_end_probs"][0].data.numpy()
        output_best_span_probs = output_dict["best_span_probs"][0].data.numpy()
        assert_almost_equal(numpy.sum(output_span_start_probs, -1), 1, decimal=6)
        assert_almost_equal(numpy.sum(output_span_end_probs, -1), 1, decimal=6)
        assert output_best_span_probs > 0 and numpy.sum(output_best_span_probs, -1) <= 1
        span_start_probs = torch.nn.functional.softmax(output_dict["span_start_logits"], dim=-1)[
            0
        ].data.numpy()
        span_end_probs = torch.nn.functional.softmax(output_dict["span_end_logits"], dim=-1)[
            0
        ].data.numpy()
        best_span_probs = (
            torch.nn.functional.softmax(output_dict["best_span_scores"], dim=-1)[0].data.numpy(),
            0,
        )
        assert_almost_equal(numpy.sum(span_start_probs, -1), 1, decimal=6)
        assert_almost_equal(numpy.sum(span_end_probs, -1), 1, decimal=6)
        assert numpy.sum(best_span_probs, -1) > 0 and numpy.sum(best_span_probs, -1) <= 1
        span_start, span_end = tuple(output_dict["best_span"][0].data.numpy())
        assert span_start >= -1
        assert span_start <= span_end
        assert span_end < self.instances[0].fields["question_with_context"].sequence_length()
        assert isinstance(output_dict["best_span_str"][0], str)


class TransformerQaV2Test(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "rc" / "transformer_qa" / "experiment_v2.jsonnet",
            FIXTURES_ROOT / "rc" / "squad2.json",
        )

    def test_model_can_train_save_and_load(self):
        # Huggingface transformer models come with pooler weights, but this model doesn't use the pooler.
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            gradients_to_ignore={
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.weight",
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.bias",
            },
        )


@requires_gpu
class TransformerQaMixedPrecisionTest(AllenNlpTestCase):
    def test_model_can_train_save_and_load_with_mixed_precision(self):
        train_model_from_file(
            FIXTURES_ROOT / "rc" / "transformer_qa" / "experiment.jsonnet",
            self.TEST_DIR,
            overrides="{'trainer.use_amp':true,'trainer.cuda_device':0}",
        )
