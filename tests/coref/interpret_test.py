import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, GatedSum
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator

from allennlp_models.coref.conll_coref_scores import ConllCorefScores
from allennlp_models.coref.mention_recall_metric import MentionRecall

logger = logging.getLogger(__name__)




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
        predictor = Predictor.from_archive(archive, "coreference-resolution")
        interpreter = SimpleGradient(predictor)
        interpretation = interpreter.saliency_interpret_from_json(inputs)
        assert interpretation is not None
        assert "instance_1" in interpretation
        assert "grad_input_1" in interpretation["instance_1"]
        grad_input_1 = interpretation["instance_1"]["grad_input_1"]
        assert len(grad_input_1) == 22  # 22 wordpieces in input

        # two interpretations should be identical for gradient
        repeat_interpretation = interpreter.saliency_interpret_from_json(inputs)
        repeat_grad_input_1 = repeat_interpretation["instance_1"]["grad_input_1"]
        for grad, repeat_grad in zip(grad_input_1, repeat_grad_input_1):
            assert grad == approx(repeat_grad)
