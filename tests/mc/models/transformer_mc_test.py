from flaky import flaky
from allennlp.commands.train import train_model_from_file
from allennlp.common.testing import ModelTestCase, AllenNlpTestCase, requires_gpu
from allennlp.data import Batch
from tests import FIXTURES_ROOT
import pytest

import allennlp_models.mc.models


class TransformerMcTest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            FIXTURES_ROOT / "mc" / "transformer_mc" / "experiment.jsonnet",
            FIXTURES_ROOT / "mc" / "piqa.jsonl",
        )

    def test_model_can_train_save_and_load(self):
        # While the model uses a pooler, it does not use the pooler that comes with the token embedder.
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            gradients_to_ignore={
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.weight",
                "_text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.bias",
                # Due to numerical instability, this scalar tensor might sometimes
                # have zero gradient.
                "_linear_layer.bias",
            },
        )

    @flaky(max_runs=3)
    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)

        # The following asserts assume that we get a fair mix of answers, some 0, some 1, some correct, and some
        # incorrect. If the model was completely un-initialized, the chance of these checks failing randomly is
        # 1/1024, and there are three of them. But the model is not completely uninitialized (in fact, it contains
        # no random weights), so we know these asserts pass. We still mark the test as flaky because random
        # drop-out could mess things up.

        assert output_dict["best_alternative"].min() == 0
        assert output_dict["best_alternative"].max() == 1

        metrics = self.model.get_metrics(reset=True)
        assert metrics["acc"] > 0


@requires_gpu
class TransformerMcMixedPrecisionTest(AllenNlpTestCase):
    def test_model_can_train_save_and_load_with_mixed_precision(self):
        train_model_from_file(
            FIXTURES_ROOT / "mc" / "transformer_mc" / "experiment.jsonnet",
            self.TEST_DIR,
            overrides="{'trainer.use_amp':true,'trainer.cuda_device':0}",
        )
