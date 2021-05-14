from allennlp.common.lazy import Lazy
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Batch, Vocabulary
from allennlp.data.image_loader import TorchImageLoader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.vision.grid_embedder import NullGridEmbedder
from allennlp.modules.vision.region_detector import RandomRegionDetector

from tests import FIXTURES_ROOT


class TestFlickr30kReader(AllenNlpTestCase):
    def setup_method(self):
        from allennlp_models.vision.dataset_readers.flickr30k import Flickr30kReader

        super().setup_method()
        self.reader = Flickr30kReader(
            image_dir=FIXTURES_ROOT / "vision" / "images" / "flickr30k",
            image_loader=TorchImageLoader(),
            image_featurizer=Lazy(NullGridEmbedder),
            data_dir=FIXTURES_ROOT / "vision" / "flickr30k" / "sentences",
            region_detector=Lazy(RandomRegionDetector),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
            is_test=True,
        )

    def test_read(self):
        instances = list(self.reader.read("test_fixtures/vision/flickr30k/test.txt"))
        assert len(instances) == 25

        instance = instances[5]
        assert len(instance.fields) == 7
        assert len(instance["caption"]) == 16
        question_tokens = [t.text for t in instance["caption"]]
        assert question_tokens == [
            "A",
            "girl",
            "with",
            "brown",
            "hair",
            "sits",
            "on",
            "the",
            "edge",
            "of",
            "a",
            "cement",
            "area",
            "overlooking",
            "water",
            ".",
        ]

        batch = Batch(instances)
        batch.index_instances(Vocabulary())
        tensors = batch.as_tensor_dict()

        # (batch size, num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (25, 2, 10)

        # (batch size, num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (25, 2, 4)

        # (batch size, num boxes (fake),)
        assert tensors["box_mask"].size() == (25, 2)

        # (batch size, num_hard_negatives, num boxes (fake), num features (fake))
        assert tensors["hard_negative_features"].size() == (25, 3, 2, 10)
