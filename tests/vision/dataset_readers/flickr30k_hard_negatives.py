from allennlp.common.lazy import Lazy
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Batch, Vocabulary
from allennlp.data.image_loader import TorchImageLoader
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
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
            region_detector=Lazy(RandomRegionDetector),
            tokenizer=PretrainedTransformerTokenizer(),
            token_indexers={"tokens": PretrainedTransformerIndexer()},
        )

    def test_read_from_dir(self):
        # Test reading from multiple files in a directory
        instances = list(self.reader.read("test_fixtures/vision/gqa/question_dir/"))
        assert len(instances) == 2

        instance = instances[1]
        assert len(instance.fields) == 6
        assert len(instance["question"]) == 10
        question_tokens = [t.text for t in instance["question"]]
        assert question_tokens == [
            "Does",
            "the",
            "table",
            "below",
            "the",
            "water",
            "look",
            "wooden",
            "and",
            "round?",
        ]
        assert instance["labels"][0].label == "yes"

        batch = Batch(instances)
        batch.index_instances(Vocabulary())
        tensors = batch.as_tensor_dict()

        # (batch size, num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (2, 2, 10)

        # (batch size, num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (2, 2, 4)

        # (batch size, num boxes (fake),)
        assert tensors["box_mask"].size() == (2, 2)
