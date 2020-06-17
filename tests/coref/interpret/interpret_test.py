from pytest import approx

from allennlp.common.testing import AllenNlpTestCase
from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from tests import FIXTURES_ROOT


class TestInterpret(AllenNlpTestCase):
    def test_simple_gradient_coref(self):
        inputs = {
            "document": "This is a single string document about a test. Sometimes it "
            "contains coreferent parts."
        }
        archive = load_archive(FIXTURES_ROOT / "coref" / "serialization" / "model.tar.gz")
        predictor = Predictor.from_archive(archive, "coreference_resolution")
        interpreter = SimpleGradient(predictor)
        interpretation = interpreter.saliency_interpret_from_json(inputs)
        assert interpretation is not None
        assert "instance_1" in interpretation
        assert "grad_input_1" in interpretation["instance_1"]
        grad_input_1 = interpretation["instance_1"]["grad_input_1"]

        # There are 16 tokens in the input.  This gets translated into 22 wordpieces, but we need to
        # compute gradients on whatever the model considered its "input", which in this case is
        # tokens, because the model uses a mismatched tokenizer / embedder.
        assert len(grad_input_1) == 16

        # two interpretations should be identical for gradient
        repeat_interpretation = interpreter.saliency_interpret_from_json(inputs)
        repeat_grad_input_1 = repeat_interpretation["instance_1"]["grad_input_1"]
        for grad, repeat_grad in zip(grad_input_1, repeat_grad_input_1):
            assert grad == approx(repeat_grad)
