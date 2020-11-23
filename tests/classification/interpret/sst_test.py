import pytest


def test_gradient_visualization():
    from allennlp.predictors.predictor import Predictor

    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz"
    )
    sentence = "a very well-made, funny and entertaining picture."

    inputs = {"sentence": sentence}
    from allennlp.interpret.saliency_interpreters import SimpleGradient

    simple_gradient_interpreter = SimpleGradient(predictor)
    simple_gradient_interpretation = simple_gradient_interpreter.saliency_interpret_from_json(
        inputs
    )

    gradients = simple_gradient_interpretation["instance_1"]["grad_input_1"]
    assert max(gradients) - min(gradients) < 0.75
