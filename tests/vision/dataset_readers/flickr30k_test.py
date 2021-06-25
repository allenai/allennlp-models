from allennlp.common.lazy import Lazy
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Batch, Vocabulary
from allennlp.data.image_loader import TorchImageLoader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp_models.vision.dataset_readers.flickr30k import Flickr30kReader
from allennlp.modules.vision.grid_embedder import NullGridEmbedder
from allennlp.modules.vision.region_detector import RandomRegionDetector

import random
from tests import FIXTURES_ROOT


class TestFlickr30kReader(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

    def test_train_read(self):
        self.reader = Flickr30kReader(
            image_dir=FIXTURES_ROOT / "vision" / "images" / "flickr30k",
            image_loader=TorchImageLoader(),
            image_featurizer=Lazy(NullGridEmbedder),
            data_dir=FIXTURES_ROOT / "vision" / "flickr30k" / "sentences",
            region_detector=Lazy(RandomRegionDetector),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
            featurize_captions=False,
            num_potential_hard_negatives=4,
        )

        instances = list(self.reader.read("test_fixtures/vision/flickr30k/test.txt"))
        assert len(instances) == 25

        instance = instances[5]
        assert len(instance.fields) == 5
        assert len(instance["caption"]) == 4
        assert len(instance["caption"][0]) == 12  # 16
        assert instance["caption"][0] != instance["caption"][1]
        assert instance["caption"][0] == instance["caption"][2]
        assert instance["caption"][0] == instance["caption"][3]
        question_tokens = [t.text for t in instance["caption"][0]]
        assert question_tokens == [
            "girl",
            "with",
            "brown",
            "hair",
            "sits",
            "on",
            "edge",
            "of",
            "concrete",
            "area",
            "overlooking",
            "water",
        ]

        batch = Batch(instances)
        batch.index_instances(Vocabulary())
        tensors = batch.as_tensor_dict()

        # (batch size, num images (3 hard negatives + gold image), num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (25, 4, 2, 10)

        # (batch size, num images (3 hard negatives + gold image), num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (25, 4, 2, 4)

        # (batch size, num images (3 hard negatives + gold image), num boxes (fake),)
        assert tensors["box_mask"].size() == (25, 4, 2)

        # (batch size)
        assert tensors["label"].size() == (25,)

    def test_evaluation_read(self):
        self.reader = Flickr30kReader(
            image_dir=FIXTURES_ROOT / "vision" / "images" / "flickr30k",
            image_loader=TorchImageLoader(),
            image_featurizer=Lazy(NullGridEmbedder),
            data_dir=FIXTURES_ROOT / "vision" / "flickr30k" / "sentences",
            region_detector=Lazy(RandomRegionDetector),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
            featurize_captions=False,
            is_evaluation=True,
            num_potential_hard_negatives=4,
        )

        instances = list(self.reader.read("test_fixtures/vision/flickr30k/test.txt"))
        assert len(instances) == 25

        instance = instances[5]
        assert len(instance.fields) == 5
        assert len(instance["caption"]) == 5
        assert len(instance["caption"][0]) == 12
        question_tokens = [t.text for t in instance["caption"][0]]
        assert question_tokens == [
            "girl",
            "with",
            "brown",
            "hair",
            "sits",
            "on",
            "edge",
            "of",
            "concrete",
            "area",
            "overlooking",
            "water",
        ]

        batch = Batch(instances)
        batch.index_instances(Vocabulary())
        tensors = batch.as_tensor_dict()

        # (batch size, num images (total), num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (25, 5, 2, 10)

        # (batch size, num images (total), num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (25, 5, 2, 4)

        # (batch size, num images (total), num boxes (fake),)
        assert tensors["box_mask"].size() == (25, 5, 2)

        # (batch size)
        assert tensors["label"].size() == (25,)
