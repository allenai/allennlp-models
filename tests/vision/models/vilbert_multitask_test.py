from allennlp.common.testing import ModelTestCase

from tests import FIXTURES_ROOT


class TestVilbertMultitask(ModelTestCase):
    def test_predict(self):
        from allennlp.models import load_archive
        from allennlp.predictors import Predictor
        import allennlp_models.vision

        archive = load_archive(FIXTURES_ROOT / "vision" / "vilbert_multitask" / "model.tar.gz")
        predictor = Predictor.from_archive(archive)

        with open(
            FIXTURES_ROOT / "vision" / "vilbert_multitask" / "dataset.json", "r"
        ) as file_input:
            json_input = [predictor.load_line(line) for line in file_input if not line.isspace()]
        predictions = predictor.predict_batch_json(json_input)
        assert all(
            "gqa_best_answer" in p or "vqa_best_answer" in p or "ve_entailment_answer" in p
            for p in predictions
        )

    def test_model_can_train_save_and_load_small_model(self):
        param_file = FIXTURES_ROOT / "vision" / "vilbert_multitask" / "experiment.jsonnet"

        # The VQA weights are going to be zero because the last batch is Visual Entailment only,
        # and so the gradients for VQA don't get set.
        self.ensure_model_can_train_save_and_load(
            param_file,
            gradients_to_ignore={"_heads.vqa.classifier.bias", "_heads.vqa.classifier.weight"},
        )
