import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.lazy import Lazy
from allennlp.data import Batch, Vocabulary
from allennlp.data.image_loader import TorchImageLoader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.vision.grid_embedder import NullGridEmbedder
from allennlp.modules.vision.region_detector import RandomRegionDetector

from tests import FIXTURES_ROOT


class TestVGQAReader(AllenNlpTestCase):
    def test_read(self):
        from allennlp_models.vision.dataset_readers.vgqa import VGQAReader

        reader = VGQAReader(
            image_dir=FIXTURES_ROOT / "vision" / "images" / "vgqa",
            image_loader=TorchImageLoader(),
            image_featurizer=Lazy(NullGridEmbedder),
            region_detector=Lazy(RandomRegionDetector),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )
        instances = list(reader.read("test_fixtures/vision/vgqa/question_answers.json"))
        assert len(instances) == 8

        instance = instances[0]
        assert len(instance.fields) == 6
        assert len(instance["question"]) == 5
        question_tokens = [t.text for t in instance["question"]]
        assert question_tokens == ["What", "is", "on", "the", "curtains?"]
        assert len(instance["labels"]) == 1
        labels = [field.label for field in instance["labels"].field_list]
        assert labels == ["sailboats"]

        batch = Batch(instances)
        batch.index_instances(Vocabulary())
        tensors = batch.as_tensor_dict()

        # (batch size, num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (8, 2, 10)

        # (batch size, num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (8, 2, 4)

        # (batch size, num boxes (fake))
        assert tensors["box_mask"].size() == (8, 2)

        # Nothing should be masked out since the number of fake boxes is the same
        # for each item in the batch.
        assert tensors["box_mask"].all()
