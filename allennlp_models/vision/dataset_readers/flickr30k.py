from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    Union,
    Optional,
    Set,
    Tuple,
    Iterable,
    List,
)
import json
import os
import heapq

from overrides import overrides
import torch
from torch import Tensor
import transformers
from random import sample, choices, randint

from allennlp.common.file_utils import cached_path
from allennlp.common.lazy import Lazy
from allennlp.common import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, TextField, TensorField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector
from allennlp_models.vision.dataset_readers.vision_reader import VisionReader

# TODO: Things slowing this down
# 3. Not being able to load hard negatives (json file?)


# TODO: implement this, filter based on that one paper (use vocab)
def filter_caption(caption):
    return caption


# Borrowed
# parse caption file for a given image
def get_caption_data(filename):
    with open(filename, "r") as f:
        captions = f.read().split("\n")

    image_id = filename[filename.rfind("/") + 1 :].split(".")[0]
    result_captions = []
    for caption in captions:
        if not caption:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
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

        result_captions.append(filter_caption(" ".join(words)))

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
    """

    def __init__(
        self,
        image_dir: Union[str, PathLike],
        *,
        image_loader: Optional[ImageLoader] = None,
        image_featurizer: Optional[Lazy[GridEmbedder]] = None,
        region_detector: Optional[Lazy[RegionDetector]] = None,
        answer_vocab: Optional[Union[str, Vocabulary]] = None,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        data_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
        write_to_cache: bool = True,
        is_test: bool = False,
        is_evaluation: bool = False,
        n: int = 100,
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
        )
        self.data_dir = data_dir
        self.is_test = is_test
        self.is_evaluation = is_evaluation
        self.n = n

        if not is_test:
            self.model = transformers.AutoModel.from_pretrained("bert-large-uncased").to(
                self.cuda_device
            )
            self.model.eval()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-large-uncased")

        # read answer vocab
        if answer_vocab is None:
            self.answer_vocab = None
        else:
            if isinstance(answer_vocab, str):
                answer_vocab = cached_path(answer_vocab, extract_archive=True)
                answer_vocab = Vocabulary.from_files(answer_vocab)
            self.answer_vocab = frozenset(
                answer_vocab.get_token_to_index_vocabulary("answers").keys()
            )

    @overrides
    def _read(self, file_path: str):
        # Plan:
        # Training:
        # 1. Process images twice, move them both to CPU
        #       Once to read them all in together, average each one, and build a tensor sized (num_images, image_dimension)
        #       Another one that has the full (num_images, num_boxes, image_dimension) so we can save features later
        #           e.g. we can get the argmax from the first tensor, and look up the features by index in the second tensor
        # 2. Process caption embeddings in batches on GPU, move to CPU
        #       We can use the same sort of logic (but new code) to do this
        # 3. Create instances
        #       a. Find the n best potential hard negatives (L2 distance)
        #       b. Pick the 3 highest scores out of the candidates (dot product between caption and images)
        # 4. Build instance
        # Evaluation:
        # 1. Process all images and move them to GPU
        #       This will (hopefully) not copy them into each instance, and instead just have one copy
        # 2. Build instances

        file_path = cached_path(file_path, extract_archive=True)
        files_in_split = set()
        with open(file_path, "r") as f:
            for line in f:
                files_in_split.add(line.rstrip("\n"))

        captions = []
        for filename in os.listdir(self.data_dir):
            if filename.split(".")[0] in files_in_split:
                full_file_path = os.path.join(self.data_dir, filename)
                captions.append(get_caption_data(full_file_path))

        # TODO: This'll have to be changed to work better with validation on multiple GPUs
        # TODO: right now, in validation, it won't have the full set of images in every instance
        # need to change.
        caption_dicts = list(self.shard_iterable(captions))

        processed_images: Iterable[Optional[Tuple[Tensor, Tensor]]]
        if self.produce_featurized_images:
            # It would be much easier to just process one image at a time, but it's faster to process
            # them in batches. So this code gathers up instances until it has enough to fill up a batch
            # that needs processing, and then processes them all.
            filenames = [f"{caption_dict['image_id']}.jpg" for caption_dict in caption_dicts]
            try:
                processed_images = list(
                    self._process_image_paths(self.images[filename] for filename in filenames)
                )
            except KeyError as e:
                missing_id = e.args[0]
                raise KeyError(
                    missing_id,
                    f"We could not find an image with the id {missing_id}. "
                    "Because of the size of the image datasets, we don't download them automatically. "
                    "Please go to https://visualqa.org/download.html, download the datasets you need, "
                    "and set the image_dir parameter to point to your download location. This dataset "
                    "reader does not care about the exact directory structure. It finds the images "
                    "wherever they are.",
                )
        else:
            processed_images = [None for _ in range(len(caption_dicts))]

        features_list = []
        averaged_features_list = []
        coordinates_list = []
        for image in processed_images:
            features, coords = self.get_image_features(image)
            features_list.append(TensorField(features))
            averaged_features_list.append(torch.mean(features, dim=0))
            coordinates_list.append(TensorField(coords))

        # Shape: (num_images, num_boxes, image_dimension)
        # features_tensor = torch.stack(features_list, dim=0)
        # Shape: (num_images, num_boxes, 4)
        # coordinates_tensor = torch.stack(coordinates_list, dim=0)
        # Shape: (num_images, image_dimension)
        averaged_features = torch.stack(averaged_features_list, dim=0)

        if self.is_evaluation:
            masks_list = []
            for image_index in range(len(caption_dicts)):
                current_feature = features_list[image_index].tensor
                masks_list.append(
                    TensorField(
                        current_feature.new_ones((current_feature.shape[0],), dtype=torch.bool),
                        padding_value=False,
                        dtype=torch.bool,
                    )
                )

            for image_index in range(len(caption_dicts)):
                caption_dict = caption_dicts[image_index]
                for caption_index in range(len(caption_dict["captions"])):
                    # caption = caption_dict["captions"][caption_index]

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
            # Shape: (num_images, num_captions_per_image = 5, caption_dimension)
            caption_tensor = self.get_caption_features(caption_dicts)

            hard_negatives_cache = {}
            for image_index in range(len(caption_dicts)):
                caption_dict = caption_dicts[image_index]
                for caption_index in range(len(caption_dict["captions"])):
                    # caption = caption_dict["captions"][caption_index]
                    if image_index not in hard_negatives_cache:
                        # I think we want negative L2 distances here
                        _, indices = (
                            -torch.cdist(
                                averaged_features, averaged_features[image_index].unsqueeze(0)
                            ).squeeze(1)
                        ).topk(self.n)
                        hard_negatives_cache[image_index] = indices.tolist()

                    instance = self.text_to_instance(
                        caption_dicts=caption_dicts,
                        image_index=image_index,
                        caption_index=caption_index,
                        features_list=features_list,
                        coordinates_list=coordinates_list,
                        averaged_features=averaged_features,
                        caption_tensor=caption_tensor,
                        potential_hard_negatives=hard_negatives_cache[image_index],
                    )

                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(
        self,
        caption_dicts: Dict[str, Dict[str, Any]],
        image_index: int,
        caption_index: int,
        features_list: List[TensorField],
        coordinates_list: List[TensorField],
        masks_list: Optional[Tensor] = None,
        averaged_features: Optional[torch.Tensor] = None,
        caption_tensor: Optional[Tensor] = None,
        potential_hard_negatives: List[int] = [],
        label: int = 0,
    ):
        if self.is_evaluation:
            caption_fields = [
                TextField(
                    self._tokenizer.tokenize(caption_dicts[image_index]["captions"][caption_index]),
                    None,
                )
            ] * len(features_list)
            fields: Dict[str, Field] = {
                "caption": ListField(caption_fields),
                "box_features": ListField(features_list),
                "box_coordinates": ListField(coordinates_list),
                "box_mask": ListField(masks_list),
                "label": LabelField(label, skip_indexing=True),
            }

            return Instance(fields)

        else:
            # index_to_image_index = {}
            # hard_negative_tensors = []
            # i = 0
            # for idx in potential_hard_negatives:
            #     if idx != image_index:
            #         index_to_image_index[i] = idx
            #         hard_negative_tensors.append(averaged_features[i])
            #         i += 1

            # TODO: change from topk
            # _, indices = (
            #     torch.stack(hard_negative_tensors, dim=0)
            #     @ caption_tensor[image_index][caption_index]
            #     # ).topk(3)
            # ).topk(1)

            # 1. Correct answer
            caption_field = TextField(
                self._tokenizer.tokenize(caption_dicts[image_index]["captions"][caption_index]),
                None,
            )
            caption_fields = [caption_field]
            features = [features_list[image_index]]
            coords = [coordinates_list[image_index]]
            masks = [
                ArrayField(
                    features_list[image_index].tensor.new_ones(
                        (features_list[image_index].tensor.shape[0],), dtype=torch.bool
                    ),
                    padding_value=False,
                    dtype=torch.bool,
                )
            ]

            # 2. Correct image, random wrong caption
            while True:
                random_image_index = randint(0, len(caption_dicts) - 1)
                random_caption_index = randint(
                    0, len(caption_dicts[random_image_index]["captions"]) - 1
                )
                if random_image_index != image_index or random_caption_index != caption_index:
                    incorrect_caption = TextField(
                        self._tokenizer.tokenize(
                            caption_dicts[random_image_index]["captions"][random_caption_index]
                        ),
                        None,
                    )
                    break

            caption_fields.append(incorrect_caption)
            features.append(features_list[image_index])
            coords.append(coordinates_list[image_index])
            masks.append(
                ArrayField(
                    features_list[image_index].tensor.new_ones(
                        (features_list[image_index].tensor.shape[0],), dtype=torch.bool
                    ),
                    padding_value=False,
                    dtype=torch.bool,
                )
            )

            # 3. Random wrong image, correct caption
            while True:
                wrong_image_index = randint(0, len(features_list) - 1)
                if wrong_image_index != image_index:
                    break
            caption_fields.append(caption_field)
            features.append(features_list[wrong_image_index])
            coords.append(coordinates_list[wrong_image_index])
            masks.append(
                ArrayField(
                    features_list[wrong_image_index].tensor.new_ones(
                        (features_list[wrong_image_index].tensor.shape[0],), dtype=torch.bool
                    ),
                    padding_value=False,
                    dtype=torch.bool,
                )
            )

            # 4. Hard negative image, correct caption
            index_to_image_index = {}
            hard_negative_tensors = []
            i = 0
            for idx in potential_hard_negatives:
                if idx != image_index:
                    index_to_image_index[i] = idx
                    hard_negative_tensors.append(averaged_features[i])
                    i += 1

            hard_negative_image_index = index_to_image_index[
                torch.argmax(
                    torch.stack(hard_negative_tensors, dim=0)
                    @ caption_tensor[image_index][caption_index]
                ).item()
            ]

            caption_fields.append(caption_field)
            features.append(features_list[hard_negative_image_index])
            coords.append(coordinates_list[hard_negative_image_index])
            masks.append(
                ArrayField(
                    features_list[hard_negative_image_index].tensor.new_ones(
                        (features_list[hard_negative_image_index].tensor.shape[0],),
                        dtype=torch.bool,
                    ),
                    padding_value=False,
                    dtype=torch.bool,
                )
            )

            fields: Dict[str, Field] = {
                "caption": ListField(caption_fields),
                "box_features": ListField(features),
                "box_coordinates": ListField(coords),
                "box_mask": ListField(masks),
                "label": LabelField(label, skip_indexing=True),
            }

            return Instance(fields)

    def get_hard_negatives(
        self,
        image_index: int,
        features_tensor: Tensor,
        averaged_features: Tensor,
        coordinates_tensor: Tensor,
        n: int = 100,
    ) -> List[Tuple[Tensor, Tensor]]:
        # Calculate the top n (let's say 100) potential hard negatives
        _, indices = (averaged_features @ averaged_features[image_index]).topk(n)
        return indices

    def get_image_features(self, image):
        if isinstance(image, str):
            features, coords = next(self._process_image_paths([image], use_cache=use_cache))
        else:
            features, coords = image
        # return features.to(device=self.cuda_device), coords.to(device=self.cuda_device)
        return features, coords

    def get_caption_features(self, captions):
        if self.is_test:
            return torch.randn(len(captions), 5, 10)

        captions_as_text = [
            c
            for caption_dict in captions
            for c in caption_dict["captions"]
        ]
        captions_hash = util.hash_object(captions_as_text)
        captions_cache_file = Path(self.feature_cache_dir) / f"CaptionsCache-{captions_hash[:12]}.pt"
        if captions_cache_file.exists():
            with captions_cache_file.open("rb") as f:
                return torch.load(f, map_location=torch.device("cpu"))

        features = []
        batch_size = 64
        with torch.no_grad():
            for batch_start in range(0, len(captions_as_text), batch_size):
                batch_end = min(batch_start + batch_size, len(captions_as_text))
                batch = self.tokenizer.batch_encode_plus(
                    captions_as_text[batch_start:batch_end],
                    return_tensors="pt",
                    padding=True
                ).to(self.cuda_device)
                embeddings = self.model(**batch).pooler_output.squeeze(0)
                features.append(embeddings.cpu())
        features = torch.cat(features)
        features = features.view(len(captions), 5, -1)
        torch.save(features, captions_cache_file)
        return features

    # todo: fix
    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        for caption in instance["caption"]:
            caption.token_indexers = self._token_indexers  # type: ignore
