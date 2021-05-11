import glob
import os
from os import PathLike
from typing import Any, Dict, Union, Optional, MutableMapping

from overrides import overrides
import torch
from torch import Tensor

from allennlp.common.file_utils import cached_path, json_lines_from_file, TensorCache, logger
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TensorField, LabelField, ListField, MetadataField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector


def extract_image_features(image: Union[str, Tuple[Tensor, Tensor]], use_cache: bool):
    if isinstance(image1, str):
        features, coords = next(self._process_image_paths([image1], use_cache=use_cache))
    else:
        features, coords = image1

    return {
        "box_features": ArrayField(features),
        "box_coordinates": ArrayField(coords),
        "box_mask": ArrayField(
            features.new_ones((features.shape[0],), dtype=torch.bool),
            padding_value=False,
            dtype=torch.bool,
        ),
    }


@DatasetReader.register("nlvr2")
class Nlvr2Reader(DatasetReader):
    # TODO: update comment
    """
    Reads the NLVR2 dataset from http://lil.nlp.cornell.edu/nlvr/.
    In this task, the model is presented with two images and a sentence referring to those images.
    The task for the model is to identify whether the sentence is true or false.
    Accordingly, the instances produced by this reader contain two images, featurized into the
    fields "box_features" and "box_coordinates". In addition to that, it produces a `TextField`
    called "sentence", and a `MetadataField` called "identifier". The latter contains the question
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
    data_dir: `str`
        Path to directory containing text files for each dataset split. These files contain
        the sentences and metadata for each task instance.  If this is `None`, we will grab the
        files from the official NLVR github repository.
    feature_cache_dir: `str`, optional
        Path to a directory that will contain a cache of featurized images.
    tokenizer: `Tokenizer`, optional, defaults to `PretrainedTransformerTokenizer("bert-base-uncased")`
    token_indexers: `Dict[str, TokenIndexer]`, optional,
        defaults to`{"tokens": PretrainedTransformerIndexer("bert-base-uncased")}`
    cuda_device: `int`, optional
        Set this to run image featurization on the given GPU. By default, image featurization runs on CPU.
    max_instances: `int`, optional
        If set, the reader only returns the first `max_instances` instances, and then stops.
        This is useful for testing.
    """

    def __init__(
        self,
        image_dir: Optional[Union[str, PathLike]] = None,
        *,
        image_loader: Optional[ImageLoader] = None,
        image_featurizer: Optional[Lazy[GridEmbedder]] = None,
        region_detector: Optional[Lazy[RegionDetector]] = None,
        answer_vocab: Optional[Union[Vocabulary, str]] = None,
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

    # def __init__(
    #     self,
    #     image_dir: Union[str, PathLike],
    #     image_loader: ImageLoader,
    #     image_featurizer: GridEmbedder,
    #     region_detector: RegionDetector,
    #     *,
    #     feature_cache_dir: Optional[Union[str, PathLike]] = None,
    #     data_dir: Optional[Union[str, PathLike]] = None,
    #     tokenizer: Optional[Tokenizer] = None,
    #     token_indexers: Optional[Dict[str, TokenIndexer]] = None,
    #     cuda_device: Optional[Union[int, torch.device]] = None,
    #     max_instances: Optional[int] = None,
    # ) -> None:
    #     super().__init__(
    #         max_instances=max_instances,
    #         manual_distributed_sharding=True,
    #         manual_multi_process_sharding=True,
    #     )

        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1
        from allennlp.common.checks import check_for_gpu

        check_for_gpu(cuda_device)
        from allennlp.common.util import int_to_device

        self.cuda_device = int_to_device(cuda_device)

        # Paths to data
        if not data_dir:
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
        from tqdm import tqdm

        logger.info("Discovering images ...")
        self.images = {
            os.path.basename(filename): filename
            for filename in tqdm(
                glob.iglob(os.path.join(image_dir, "**", "*.png"), recursive=True),
                desc="Discovering images",
            )
        }
        logger.info("Done discovering images")

        # TODO: make sure these are accurate

        # tokenizers and indexers
        if not tokenizer:
            tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}
        self._token_indexers = token_indexers

        # image loading
        self.image_loader = image_loader
        self.image_featurizer = image_featurizer.to(self.cuda_device)
        self.region_detector = region_detector.to(self.cuda_device)

        # feature cache
        if feature_cache_dir is None:
            self._features_cache: MutableMapping[str, Tensor] = {}
            self._coordinates_cache: MutableMapping[str, Tensor] = {}
        else:
            os.makedirs(feature_cache_dir, exist_ok=True)
            self._features_cache = TensorCache(os.path.join(feature_cache_dir, "features"))
            self._coordinates_cache = TensorCache(os.path.join(feature_cache_dir, "coordinates"))

        # TODO: end TODO

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
                for blob in blobs:
                    image_name_base = identifier[: identifier.rindex("-")]
                    image_paths1.append(self.images[f"{image_name_base}-img0.png"])
                    image_paths2.append(self.images[f"{image_name_base}-img1.png"])
            except KeyError as e:
                missing_id = e.args[0]
                raise KeyError(
                    missing_id,
                    f"We could not find an image with the id {missing_id}. "
                    "Because of the size of the image datasets, we don't download them automatically. "
                    "Please go to https://github.com/lil-lab/nlvr/tree/master/nlvr2, download the datasets you need, "
                    "and set the image_dir parameter to point to your download location. This dataset "
                    "reader does not care about the exact directory structure. It finds the images "
                    "wherever they are.",
                )

            processed_images1 = self._process_image_paths(image_paths1)
            processed_images2 = self._process_image_paths(image_paths2)
        else:
            processed_images1 = [None for _ in range(len(blob_dicts))]
            processed_images2 = [None for _ in range(len(blob_dicts))]

        for json_blob, image1, image2 in zip(blob_dicts, processed_images1, processed_images2):
            identifier = json_blob["identifier"]
            sentence = json_blob["sentence"]
            label = bool(json_blob["label"])
            instance = self.text_to_instance(identifier, sentence, image1, image2, label)
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(
        self,
        identifier: str,
        sentence: str,
        image1: Union[str, Tuple[Tensor, Tensor]],
        image2: Union[str, Tuple[Tensor, Tensor]],
        label: bool,  # type: ignore
    ) -> Instance:
        tokenized_sentence = self._tokenizer.tokenize(sentence)

        fields = {
            "sentence": TextField(tokenized_sentence, None),
            "identifier": MetadataField(identifier),
        }

        if label is not None:
            fields["label"] = LabelField(int(label), skip_indexing=True)

        fields["image1"] = extract_image_features(image1, use_cache)
        fields["image2"] = extract_image_features(image2, use_cache)

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["sentence"].token_indexers = self._token_indexers  # type: ignore
