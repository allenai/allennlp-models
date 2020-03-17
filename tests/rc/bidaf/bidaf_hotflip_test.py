from allennlp.interpret.attackers import Hotflip
from allennlp.interpret.attackers.hotflip import DEFAULT_IGNORE_TOKENS
from allennlp.models import load_archive
from allennlp.predictors import Predictor

import allennlp_models.rc.bidaf
from tests import FIXTURES_ROOT


class TestHotflip:
    def test_using_squad_model(self):
        inputs = {
            "question": "OMG, I heard you coded a test that succeeded on its first attempt, is that true?",
            "passage": "Bro, never doubt a coding wizard! I am the king of software, MWAHAHAHA",
        }

        archive = load_archive(FIXTURES_ROOT / "rc" / "bidaf" / "serialization" / "model.tar.gz")
        predictor = Predictor.from_archive(archive, "reading-comprehension")

        hotflipper = Hotflip(predictor)
        hotflipper.initialize()
        attack = hotflipper.attack_from_json(inputs, "question", "grad_input_2")
        print(attack)
        assert attack is not None
        assert "final" in attack
        assert "original" in attack
        assert "outputs" in attack
        assert len(attack["final"][0]) == len(
            attack["original"]
        )  # hotflip replaces words without removing

        instance = predictor._json_to_instance(inputs)
        assert instance["question"] != attack["final"][0]  # check that the input has changed.

        outputs = predictor._model.forward_on_instance(instance)
        original_labeled_instance = predictor.predictions_to_labeled_instances(instance, outputs)[0]
        original_span_start = original_labeled_instance["span_start"].sequence_index
        original_span_end = original_labeled_instance["span_end"].sequence_index

        flipped_span_start = attack["outputs"]["best_span"][0]
        flipped_span_end = attack["outputs"]["best_span"][1]

        for i, token in enumerate(instance["question"]):
            token = str(token)
            if token in DEFAULT_IGNORE_TOKENS:
                assert token in attack["final"][0]  # ignore tokens should not be changed
            # HotFlip keeps changing tokens until either the prediction changes or all tokens have
            # been changed. If there are tokens in the HotFlip final result that were in the
            # original (i.e., not all tokens were flipped), then the prediction should be
            # different.
            else:
                if token == attack["final"][0][i]:
                    assert (
                        original_span_start != flipped_span_start
                        or original_span_end != flipped_span_end
                    )
