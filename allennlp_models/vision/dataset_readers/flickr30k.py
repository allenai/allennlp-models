from os import PathLike
from pathlib import Path
import logging
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)
import os
import tqdm

from overrides import overrides
import torch
from torch import Tensor
import transformers
from random import randint

from allennlp.common.file_utils import cached_path
from allennlp.common.lazy import Lazy
from allennlp.common import util
from allennlp.common.file_utils import TensorCache
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, TextField, TensorField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector

from allennlp_models.vision.dataset_readers import utils
from allennlp_models.vision.dataset_readers.vision_reader import VisionReader

logger = logging.getLogger(__name__)


# Parse caption file
def get_caption_data(filename: str):
    with open(filename, "r") as f:
        captions = f.read().split("\n")

    image_id = os.path.splitext(os.path.basename(filename))[0]
    result_captions = []
    for caption in captions:
        if not caption:
            continue

        words = []
        add_to_phrase = False
        for token in caption.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                else:
                    words.append(token)

        result_captions.append(utils.preprocess_answer(" ".join(words)))

    caption_data = {"image_id": image_id, "captions": result_captions}
    return caption_data


@DatasetReader.register("flickr30k")
class Flickr30kReader(VisionReader):
    """
    Parameters
    ----------
    image_dir: `str`
        Path to directory containing `png` image files.
    image_loader : `ImageLoader`
    image_featurizer: `Lazy[GridEmbedder]`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `Lazy[RegionDetector]`
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    data_dir: `str`
        Path to directory containing text files for each dataset split. These files contain
        the captions and metadata for each task instance.
    tokenizer: `Tokenizer`, optional
    token_indexers: `Dict[str, TokenIndexer]`
    featurize_captions: `bool`, optional
        If we should featurize captions while calculating hard negatives, or use placeholder features.
    is_evaluation: `bool`, optional
        If the reader should return instances for evaluation or training.
    num_potential_hard_negatives: int, optional
        The number of potential hard negatives to consider.
    """

    def __init__(
        self,
        image_dir: Union[str, PathLike],
        *,
        image_loader: Optional[ImageLoader] = None,
        image_featurizer: Optional[Lazy[GridEmbedder]] = None,
        region_detector: Optional[Lazy[RegionDetector]] = None,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        data_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
        write_to_cache: bool = True,
        featurize_captions: bool = True,
        is_evaluation: bool = False,
        num_potential_hard_negatives: int = 100,
    ) -> None:
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
            manual_distributed_sharding=False,
            manual_multiprocess_sharding=False,
        )
        self.data_dir = cached_path(data_dir, extract_archive=True)
        self.featurize_captions = featurize_captions
        self.is_evaluation = is_evaluation
        self.num_potential_hard_negatives = num_potential_hard_negatives

        if self.featurize_captions:
            self.model = transformers.AutoModel.from_pretrained("bert-large-uncased").to(
                self.cuda_device
            )
            self.model.eval()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-large-uncased")

        # feature cache
        self.hard_negative_features_cache_dir = feature_cache_dir
        self.hard_negative_coordinates_cache_dir = feature_cache_dir
        self._hard_negative_features_cache_instance: Optional[MutableMapping[str, Tensor]] = None
        self._hard_negative_coordinates_cache_instance: Optional[MutableMapping[str, Tensor]] = None
        if self.hard_negative_features_cache_dir and self.hard_negative_coordinates_cache_dir:
            logger.info(f"Calculating hard negatives with a cache at {self.feature_cache_dir}")

    @property
    def _hard_negative_features_cache(self) -> MutableMapping[str, Tensor]:
        if self._hard_negative_features_cache_instance is None:
            if self.hard_negative_features_cache_dir is None:
                logger.info("could not find feature cache dir")
                self._hard_negative_features_cache_instance = {}
            else:
                logger.info("found feature cache dir")
                os.makedirs(self.feature_cache_dir, exist_ok=True)  # type: ignore
                self._hard_negative_features_cache_instance = TensorCache(
                    os.path.join(self.feature_cache_dir, "hard_negative_features"),  # type: ignore
                    read_only=not self.write_to_cache,
                )

        return self._hard_negative_features_cache_instance

    @property
    def _hard_negative_coordinates_cache(self) -> MutableMapping[str, Tensor]:
        if self._hard_negative_coordinates_cache_instance is None:
            if self.hard_negative_coordinates_cache_dir is None:
                self._hard_negative_coordinates_cache_instance = {}
            else:
                os.makedirs(self.feature_cache_dir, exist_ok=True)  # type: ignore
                self._hard_negative_coordinates_cache_instance = TensorCache(
                    os.path.join(self.feature_cache_dir, "hard_negative_coordinates"),  # type: ignore
                    read_only=not self.write_to_cache,
                )

        return self._hard_negative_coordinates_cache_instance

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path, extract_archive=True)
        files_in_split = set()
        i = 0
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if self.max_instances is not None and i * 5 >= self.max_instances:
                    break
                files_in_split.add(line.rstrip("\n"))

        caption_dicts = []
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.split(".")[0] in files_in_split:
                full_file_path = os.path.join(self.data_dir, filename)
                caption_dicts.append(get_caption_data(full_file_path))

        processed_images: Iterable[
            Optional[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]]
        ]
        filenames = [f"{caption_dict['image_id']}.jpg" for caption_dict in caption_dicts]
        try:
            processed_images = self._process_image_paths(
                self.images[filename] for filename in tqdm.tqdm(filenames, desc="Processing images")
            )
        except KeyError as e:
            missing_id = e.args[0]
            raise KeyError(
                missing_id,
                f"We could not find an image with the id {missing_id}. "
                "Because of the size of the image datasets, we don't download them automatically. "
                "Please go to https://shannon.cs.illinois.edu/DenotationGraph/, download the datasets you need, "
                "and set the image_dir parameter to point to your download location. This dataset "
                "reader does not care about the exact directory structure. It finds the images "
                "wherever they are.",
            )

        features_list = []
        averaged_features_list = []
        coordinates_list = []
        masks_list = []
        for features, coords, _, _ in processed_images:
            features_list.append(TensorField(features))
            averaged_features_list.append(torch.mean(features, dim=0))
            coordinates_list.append(TensorField(coords))
            masks_list.append(
                ArrayField(
                    features.new_ones((features.shape[0],), dtype=torch.bool),
                    padding_value=False,
                    dtype=torch.bool,
                )
            )

        # Validation instances are a 1000-way multiple choice,
        # one for each image in the validation set.
        if self.is_evaluation:
            for image_index in range(len(caption_dicts)):
                caption_dict = caption_dicts[image_index]
                for caption_index in range(len(caption_dict["captions"])):
                    instance = self.text_to_instance(
                        caption_dicts=caption_dicts,
                        image_index=image_index,
                        caption_index=caption_index,
                        features_list=features_list,
                        coordinates_list=coordinates_list,
                        masks_list=masks_list,
                        label=image_index,
                    )

                    if instance is not None:
                        yield instance
        else:
            # Shape: (num_images, image_dimension)
            averaged_features = torch.stack(averaged_features_list, dim=0)
            del averaged_features_list

            # Shape: (num_images, num_captions_per_image = 5, caption_dimension)
            caption_tensor = self.get_caption_features(caption_dicts)

            for image_index, caption_dict in enumerate(caption_dicts):
                for caption_index in range(len(caption_dict["captions"])):
                    hard_negative_features, hard_negative_coordinates = self.get_hard_negatives(
                        image_index,
                        caption_index,
                        caption_dicts,
                        averaged_features,
                        features_list,
                        coordinates_list,
                        caption_tensor,
                    )

                    instance = self.text_to_instance(
                        caption_dicts=caption_dicts,
                        image_index=image_index,
                        caption_index=caption_index,
                        features_list=features_list,
                        coordinates_list=coordinates_list,
                        masks_list=masks_list,
                        hard_negative_features=hard_negative_features,
                        hard_negative_coordinates=hard_negative_coordinates,
                    )

                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(
        self,
        caption_dicts: List[Dict[str, Any]],
        image_index: int,
        caption_index: int,
        features_list: List[TensorField] = [],
        coordinates_list: List[TensorField] = [],
        masks_list: List[TensorField] = [],
        hard_negative_features: Optional[Tensor] = None,
        hard_negative_coordinates: Optional[Tensor] = None,
        label: int = 0,
    ):
        if self.is_evaluation:
            caption_fields = [
                TextField(
                    self._tokenizer.tokenize(caption_dicts[image_index]["captions"][caption_index]),
                    None,
                )
            ] * len(caption_dicts)

            return Instance(
                {
                    "caption": ListField(caption_fields),
                    "box_features": ListField(features_list),
                    "box_coordinates": ListField(coordinates_list),
                    "box_mask": ListField(masks_list),
                    "label": LabelField(label, skip_indexing=True),
                }
            )

        else:
            # 1. Correct answer
            caption_field = TextField(
                self._tokenizer.tokenize(caption_dicts[image_index]["captions"][caption_index]),
                None,
            )
            caption_fields = [caption_field]
            features = [features_list[image_index]]
            coords = [coordinates_list[image_index]]
            masks = [masks_list[image_index]]

            # 2. Correct image, random wrong caption
            random_image_index = randint(0, len(caption_dicts) - 2)
            if random_image_index == image_index:
                random_image_index += 1
            random_caption_index = randint(0, 4)

            caption_fields.append(
                TextField(
                    self._tokenizer.tokenize(
                        caption_dicts[random_image_index]["captions"][random_caption_index]
                    ),
                    None,
                )
            )
            features.append(features_list[image_index])
            coords.append(coordinates_list[image_index])
            masks.append(masks_list[image_index])

            # 3. Random wrong image, correct caption
            wrong_image_index = randint(0, len(features_list) - 2)
            if wrong_image_index == image_index:
                wrong_image_index += 1

            caption_fields.append(caption_field)
            features.append(features_list[wrong_image_index])
            coords.append(coordinates_list[wrong_image_index])
            masks.append(masks_list[wrong_image_index])

            # 4. Hard negative image, correct caption
            caption_fields.append(caption_field)
            features.append(TensorField(hard_negative_features))
            coords.append(TensorField(hard_negative_coordinates))
            masks.append(
                ArrayField(
                    hard_negative_features.new_ones(
                        (hard_negative_features.shape[0],),
                        dtype=torch.bool,
                    ),
                    padding_value=False,
                    dtype=torch.bool,
                )
            )

            return Instance(
                {
                    "caption": ListField(caption_fields),
                    "box_features": ListField(features),
                    "box_coordinates": ListField(coords),
                    "box_mask": ListField(masks),
                    "label": LabelField(label, skip_indexing=True),
                }
            )

    def get_hard_negatives(
        self,
        image_index: int,
        caption_index: int,
        caption_dicts: List[Dict[str, Any]],
        averaged_features: Tensor,
        features_list: List[TensorField],
        coordinates_list: List[TensorField],
        caption_tensor: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        image_id = caption_dicts[image_index]["image_id"]
        caption = caption_dicts[image_index]["captions"][caption_index]
        cache_id = f"{image_id}-{util.hash_object(caption)}"

        if (
            cache_id not in self._hard_negative_features_cache
            or cache_id not in self._hard_negative_coordinates_cache
        ):
            _, indices = (
                -torch.cdist(
                    averaged_features, averaged_features[image_index].unsqueeze(0)
                ).squeeze(1)
            ).topk(min(averaged_features.size(0), self.num_potential_hard_negatives))

            index_to_image_index = {}
            hard_negative_tensors = []
            i = 0
            for idx in indices.tolist():
                if idx != image_index:
                    index_to_image_index[i] = idx  #
                    hard_negative_tensors.append(averaged_features[i])
                    i += 1

            hard_negative_image_index = index_to_image_index[
                torch.argmax(
                    torch.stack(hard_negative_tensors, dim=0)
                    @ caption_tensor[image_index][caption_index]
                ).item()
            ]

            self._hard_negative_features_cache[cache_id] = features_list[
                hard_negative_image_index
            ].tensor
            self._hard_negative_coordinates_cache[cache_id] = coordinates_list[
                hard_negative_image_index
            ].tensor

        return (
            self._hard_negative_features_cache[cache_id],
            self._hard_negative_coordinates_cache[cache_id],
        )

    def get_caption_features(self, captions):
        if not self.featurize_captions:
            return torch.ones(len(captions), 5, 10)

        captions_as_text = [c for caption_dict in captions for c in caption_dict["captions"]]
        if self.feature_cache_dir is not None:
            captions_hash = util.hash_object(captions_as_text)
            captions_cache_file = (
                Path(self.feature_cache_dir) / f"CaptionsCache-{captions_hash[:12]}.pt"
            )
            if captions_cache_file.exists():
                with captions_cache_file.open("rb") as f:
                    return torch.load(f, map_location=torch.device("cpu"))

        features = []
        batch_size = 64
        with torch.no_grad():
            for batch_start in tqdm.trange(
                0, len(captions_as_text), batch_size, desc="Featurizing captions"
            ):
                batch_end = min(batch_start + batch_size, len(captions_as_text))
                batch = self.tokenizer.batch_encode_plus(
                    captions_as_text[batch_start:batch_end], return_tensors="pt", padding=True
                ).to(self.cuda_device)
                embeddings = self.model(**batch).pooler_output.squeeze(0)
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)
                features.append(embeddings.cpu())
        features = torch.cat(features)
        features = features.view(len(captions), 5, -1)
        if self.feature_cache_dir is not None:
            temp_captions_cache_file = captions_cache_file.with_suffix(".tmp")
            try:
                torch.save(features, temp_captions_cache_file)
                temp_captions_cache_file.replace(captions_cache_file)
            finally:
                try:
                    temp_captions_cache_file.unlink()
                except FileNotFoundError:
                    pass
            return features

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        for caption in instance["caption"]:
            caption.token_indexers = self._token_indexers  # type: ignore
