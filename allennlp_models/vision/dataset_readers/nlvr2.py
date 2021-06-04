import logging
from os import PathLike
from typing import Any, Dict, Iterable, Tuple, Union, Optional

from overrides import overrides
import torch
from torch import Tensor

from allennlp.common.file_utils import cached_path, json_lines_from_file
from allennlp.common.lazy import Lazy
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, MetadataField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector

from allennlp_models.vision.dataset_readers.vision_reader import VisionReader

logger = logging.getLogger(__name__)


@DatasetReader.register("nlvr2")
class Nlvr2Reader(VisionReader):
    """
    Reads the NLVR2 dataset from [http://lil.nlp.cornell.edu/nlvr/](http://lil.nlp.cornell.edu/nlvr/).
    In this task, the model is presented with two images and a hypothesis referring to those images.
    The task for the model is to identify whether the hypothesis is true or false.
    Accordingly, the instances produced by this reader contain two images, featurized into the
    fields "box_features" and "box_coordinates". In addition to that, it produces a `TextField`
    called "hypothesis", and a `MetadataField` called "identifier". The latter contains the question
    id from the question set.

    Parameters
    ----------
    image_dir: `str`
        Path to directory containing `png` image files.
    image_loader: `ImageLoader`
        An image loader to read the images with
    image_featurizer: `GridEmbedder`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `RegionDetector`
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    feature_cache_dir: `str`, optional
        If given, the reader will attempt to use the featurized image cache in this directory.
        Caching the featurized images can result in big performance improvements, so it is
        recommended to set this.
    tokenizer: `Tokenizer`, optional, defaults to `PretrainedTransformerTokenizer("bert-base-uncased")`
    token_indexers: `Dict[str, TokenIndexer]`, optional,
        defaults to`{"tokens": PretrainedTransformerIndexer("bert-base-uncased")}`
    cuda_device: `int`, optional
        Set this to run image featurization on the given GPU. By default, image featurization runs on CPU.
    max_instances: `int`, optional
        If set, the reader only returns the first `max_instances` instances, and then stops.
        This is useful for testing.
    image_processing_batch_size: `int`
        The number of images to process at one time while featurizing. Default is 8.
    """

    def __init__(
        self,
        image_dir: Optional[Union[str, PathLike]] = None,
        *,
        image_loader: Optional[ImageLoader] = None,
        image_featurizer: Optional[Lazy[GridEmbedder]] = None,
        region_detector: Optional[Lazy[RegionDetector]] = None,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
        write_to_cache: bool = True,
    ) -> None:
        run_featurization = image_loader and image_featurizer and region_detector
        if image_dir is None and run_featurization:
            raise ValueError(
                "Because of the size of the image datasets, we don't download them automatically. "
                "Please go to https://github.com/lil-lab/nlvr/tree/master/nlvr2, download the datasets you need, "
                "and set the image_dir parameter to point to your download location. This dataset "
                "reader does not care about the exact directory structure. It finds the images "
                "wherever they are."
            )

        super().__init__(
            image_dir,
            image_loader=image_loader,
            image_featurizer=image_featurizer,
            region_detector=region_detector,
            feature_cache_dir=feature_cache_dir,
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            cuda_device=cuda_device,
            max_instances=max_instances,
            image_processing_batch_size=image_processing_batch_size,
            write_to_cache=write_to_cache,
        )

        github_url = "https://raw.githubusercontent.com/lil-lab/nlvr/"
        nlvr_commit = "68a11a766624a5b665ec7594982b8ecbedc728c7"
        data_dir = f"{github_url}{nlvr_commit}/nlvr2/data"
        self.splits = {
            "dev": f"{data_dir}/dev.json",
            "test": f"{data_dir}/test1.json",
            "train": f"{data_dir}/train.json",
            "balanced_dev": f"{data_dir}/balanced/balanced_dev.json",
            "balanced_test": f"{data_dir}/balanced/balanced_test1.json",
            "unbalanced_dev": f"{data_dir}/balanced/unbalanced_dev.json",
            "unbalanced_test": f"{data_dir}/balanced/unbalanced_test1.json",
        }

    @overrides
    def _read(self, split_or_filename: str):
        filename = self.splits.get(split_or_filename, split_or_filename)

        json_file_path = cached_path(filename)

        blobs = []
        json_blob: Dict[str, Any]
        for json_blob in json_lines_from_file(json_file_path):
            blobs.append(json_blob)

        blob_dicts = list(self.shard_iterable(blobs))
        processed_images1: Iterable[Optional[Tuple[Tensor, Tensor]]]
        processed_images2: Iterable[Optional[Tuple[Tensor, Tensor]]]
        if self.produce_featurized_images:
            # It would be much easier to just process one image at a time, but it's faster to process
            # them in batches. So this code gathers up instances until it has enough to fill up a batch
            # that needs processing, and then processes them all.

            try:
                image_paths1 = []
                image_paths2 = []
                for blob in blob_dicts:
                    identifier = blob["identifier"]
                    image_name_base = identifier[: identifier.rindex("-")]
                    image_paths1.append(self.images[f"{image_name_base}-img0.png"])
                    image_paths2.append(self.images[f"{image_name_base}-img1.png"])
            except KeyError as e:
                missing_id = e.args[0]
                raise KeyError(
                    missing_id,
                    f"We could not find an image with the id {missing_id}. "
                    "Because of the size of the image datasets, we don't download them automatically. "
                    "Please go to https://github.com/lil-lab/nlvr/tree/master/nlvr2, download the "
                    "datasets you need, and set the image_dir parameter to point to your download "
                    "location. This dataset reader does not care about the exact directory "
                    "structure. It finds the images wherever they are.",
                )

            processed_images1 = self._process_image_paths(image_paths1)
            processed_images2 = self._process_image_paths(image_paths2)
        else:
            processed_images1 = [None for _ in range(len(blob_dicts))]
            processed_images2 = [None for _ in range(len(blob_dicts))]

        attempted_instances = 0
        for json_blob, image1, image2 in zip(blob_dicts, processed_images1, processed_images2):
            identifier = json_blob["identifier"]
            hypothesis = json_blob["sentence"]
            label = json_blob["label"] == "True"
            instance = self.text_to_instance(identifier, hypothesis, image1, image2, label)
            if instance is not None:
                attempted_instances += 1
                yield instance
        logger.info(f"Successfully yielded {attempted_instances} instances")

    def extract_image_features(self, image: Union[str, Tuple[Tensor, Tensor]], use_cache: bool):
        if isinstance(image, str):
            features, coords = next(self._process_image_paths([image], use_cache=use_cache))
        else:
            features, coords = image

        return (
            ArrayField(features),
            ArrayField(coords),
            ArrayField(
                features.new_ones((features.shape[0],), dtype=torch.bool),
                padding_value=False,
                dtype=torch.bool,
            ),
        )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        identifier: Optional[str],
        hypothesis: str,
        image1: Union[str, Tuple[Tensor, Tensor]],
        image2: Union[str, Tuple[Tensor, Tensor]],
        label: bool,
        use_cache: bool = True,
    ) -> Instance:
        hypothesis_field = TextField(self._tokenizer.tokenize(hypothesis), None)
        box_features1, box_coordinates1, box_mask1 = self.extract_image_features(image1, use_cache)
        box_features2, box_coordinates2, box_mask2 = self.extract_image_features(image2, use_cache)

        fields = {
            "hypothesis": ListField([hypothesis_field, hypothesis_field]),
            "box_features": ListField([box_features1, box_features2]),
            "box_coordinates": ListField([box_coordinates1, box_coordinates2]),
            "box_mask": ListField([box_mask1, box_mask2]),
        }

        if identifier is not None:
            fields["identifier"] = MetadataField(identifier)

        if label is not None:
            fields["label"] = LabelField(int(label), skip_indexing=True)

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["hypothesis"][0].token_indexers = self._token_indexers  # type: ignore
        instance["hypothesis"][1].token_indexers = self._token_indexers  # type: ignore
