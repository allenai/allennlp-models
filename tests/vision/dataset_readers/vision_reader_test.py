from allennlp.common.lazy import Lazy
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.image_loader import TorchImageLoader
from allennlp.modules.vision.grid_embedder import NullGridEmbedder
from allennlp.modules.vision.region_detector import RandomRegionDetector
from allennlp_models.vision.dataset_readers.vision_reader import VisionReader
from tests import FIXTURES_ROOT


class TestVisionReader(AllenNlpTestCase):
    def test_load_images(self):
        reader = VisionReader(
            image_dir=FIXTURES_ROOT / "vision" / "images" / "vision_reader",
            image_loader=TorchImageLoader(),
            image_featurizer=Lazy(NullGridEmbedder),
            region_detector=Lazy(RandomRegionDetector),
        )
        assert len(reader.images) == 3
        assert set(reader.images.keys()) == {
            "png_example.png",
            "jpg_example.jpg",
            "jpeg_example.jpeg",
        }
