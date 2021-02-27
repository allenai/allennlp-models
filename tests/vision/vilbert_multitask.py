from allennlp.common.testing import ModelTestCase

from tests import FIXTURES_ROOT


class TestVilbertMultitask(ModelTestCase):
    def test_predict(self):
        from allennlp.models import load_archive
        from allennlp.predictors import Predictor
        import allennlp_models.vision

        archive = load_archive(
            "https://storage.googleapis.com/allennlp-public-models/vilbert-multitask.2021-02-14.tar.gz"
        )
        predictor = Predictor.from_archive(archive)

        with open(FIXTURES_ROOT / "vision" / "vilbert_multitask.json", "r") as file_input:
            json_input = [predictor.load_line(line) for line in file_input if not line.isspace()]
        predictions = predictor.predict_batch_json(json_input)
        assert all(
            "gqa_best_answer" in p or "vqa_best_answer" in p or "ve_entailment_answer" in p
            for p in predictions
        )
