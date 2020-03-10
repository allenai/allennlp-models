from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary

from allennlp_models.rc.transformer_qa import TransformerSquadReader
from allennlp_models.rc.transformer_qa import TransformerQA
from allennlp_models.rc.transformer_qa import TransformerQAPredictor


class TestTransformerQAPredictor(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.reader = TransformerSquadReader(length_limit=50, stride=10)
        self.vocab = Vocabulary()
        self.model = TransformerQA(self.vocab)
        self.predictor = TransformerQAPredictor(self.model, self.reader)
        # We're running an untrained model, so the answers will be random.

    def test_predict_single_instance(self):
        prediction = self.predictor.predict(
            "What is love?", "Baby don't hurt me, don't hurt me, no more."
        )
        assert all(0 <= i < len(prediction["context_tokens"]) for i in prediction["best_span"])

    def test_predict_long_instance(self):
        # We use a short context and a long context, so that the long context has to be broken into multiple
        # instances and re-assembled into a single answer.
        questions = [
            {
                "question": "Do fish drink water?",
                "context": """
                    A freshwater fish's insides has a higher salt content than the exterior water, so their bodies
                    are constantly absorbing water through osmosis via their permeable gills.
                """,
            },
            {
                "question": "Why don't animals have wheels?",
                "context": """
                    The worlds of fiction and myth are full of wheeled creatures, so why not the real world? After
                    all, the wheel is an efficient design, and it seems like there would be obvious advantages to
                    quickly moving around while consuming little energy.
                    The key is to remember that evolution is a process, not something that happens overnight. A
                    giraffe with just a little bit longer neck than the others will be able to reach slightly
                    higher trees, which will ultimately lead to the species' neck length getting longer and longer
                    over generations. In the meantime, those other giraffes can still eat, just not quite as well.
                    But a wheel either works or it doesn't. A somewhat circular semi-wheelish thing would only be a
                    hindrance, and evolution can't produce a trait that's perfect from the get-go.
                """,
            },
        ]
        predictions = self.predictor.predict_batch_json(questions)
        assert len(predictions) == 2
