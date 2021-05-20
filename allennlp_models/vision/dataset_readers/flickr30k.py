from os import PathLike
from typing import (
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
import faiss
import transformers
from random import sample, choices

from allennlp.common.file_utils import cached_path
from allennlp.common.lazy import Lazy
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
# 1. Too big of a set of images to search through
# 2. Model DRASTICALLY slows it down (probably wrong device?)
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

        # if self.is_evaluation:
        #     full_filenames = [f"{caption_dict['image_id']}.jpg" for caption_dict in captions]
        #     full_images = list(self._process_image_paths(self.images[filename] for filename in full_filenames))

        # TODO: This'll have to be changed to work better with validation on multiple GPUs
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
        coordinates_list = []
        for image in processed_images:
            features, coords = self.get_image_features(image)
            features_list.append(features)
            coordinates_list.append(coords)

        # Shape: (num_images, num_boxes, image_dimension)
        features_tensor = torch.stack(features_list, dim=0)
        # Shape: (num_images, num_boxes, 4)
        coordinates_tensor = torch.stack(coordinates_list, dim=0)
        # Shape: (num_images, image_dimension)
        averaged_features = torch.mean(features_tensor, dim=1)

        if self.is_evaluation:
            # TODO: Make mask tensor here
            masks_list = []
            for image_index in range(len(caption_dicts)):
                current_feature = features_tensor[image_index]
                masks_list.append(
                    current_feature.new_ones((current_feature.shape[0],), dtype=torch.bool)
                )
            masks_tensor = torch.stack(masks_list, dim=0)
            features_tensor = features_tensor.to(self.cuda_device)

            for image_index in range(len(caption_dicts)):
                caption_dict = caption_dicts[image_index]
                # processed_image = processed_images[image_index]
                for caption_index in range(len(caption_dict["captions"])):
                    caption = caption_dict["captions"][caption_index]

                    instance = self.text_to_instance(
                        caption=caption,
                        image_index=image_index,
                        caption_index=caption_index,
                        features_tensor=features_tensor,
                        coordinates_tensor=coordinates_tensor,
                        masks_tensor=masks_tensor,
                        label=image_index,
                    )

                    if instance is not None:
                        yield instance
        else:
            # Shape: (num_images, num_captions_per_image = 5, caption_dimension)
            caption_tensor = self.get_caption_features(captions)

            hard_negatives_cache = {}
            for image_index in range(len(caption_dicts)):
                caption_dict = caption_dicts[image_index]
                # processed_image = processed_images[image_index]
                for caption_index in range(len(caption_dict["captions"])):
                    caption = caption_dict["captions"][caption_index]
                    if image_index not in hard_negatives_cache:
                        # I think we want negative L2 distances here
                        _, indices = (
                            -torch.cdist(
                                averaged_features, averaged_features[image_index].unsqueeze(0)
                            ).squeeze(1)
                        ).topk(self.n)
                        # _, indices = (averaged_features @ averaged_features[image_index]).topk(n)
                        hard_negatives_cache[image_index] = indices.tolist()

                    instance = self.text_to_instance(
                        caption=caption,
                        image_index=image_index,
                        caption_index=caption_index,
                        features_tensor=features_tensor,
                        coordinates_tensor=coordinates_tensor,
                        averaged_features=averaged_features,
                        caption_tensor=caption_tensor,
                        potential_hard_negatives=hard_negatives_cache[image_index],
                    )

                    if instance is not None:
                        yield instance

    @overrides
    def text_to_instance(
        self,
        caption: str,
        image_index: int,
        caption_index: int,
        features_tensor: torch.Tensor,
        coordinates_tensor: torch.Tensor,
        masks_tensor: Optional[Tensor] = None,
        averaged_features: Optional[torch.Tensor] = None,
        caption_tensor: Optional[Tensor] = None,
        potential_hard_negatives: List[int] = [],
        label: int = 0,
    ):
        if self.is_evaluation:
            fields: Dict[str, Field] = {
                "caption": TextField(self._tokenizer.tokenize(caption), None),
                "box_features": TensorField(features_tensor),
                "box_coordinates": TensorField(coordinates_tensor),
                "box_mask": TensorField(masks_tensor, padding_value=False, dtype=torch.bool),
                "label": LabelField(label, skip_indexing=True),
            }

            return Instance(fields)

        else:
            index_to_image_index = {}
            hard_negative_tensors = []
            i = 0
            for idx in potential_hard_negatives:
                if idx != image_index:
                    index_to_image_index[i] = idx
                    hard_negative_tensors.append(averaged_features[i])
                    i += 1

            _, indices = (
                torch.stack(hard_negative_tensors, dim=0)
                @ caption_tensor[image_index][caption_index]
            ).topk(3)

            features = [ArrayField(features_tensor[image_index])]
            coords = [ArrayField(coordinates_tensor[image_index])]
            masks = [
                ArrayField(
                    features_tensor[image_index].new_ones(
                        (features_tensor[image_index].shape[0],), dtype=torch.bool
                    ),
                    padding_value=False,
                    dtype=torch.bool,
                )
            ]

            for idx in indices.tolist():
                hard_negative_index = index_to_image_index[idx]
                features.append(ArrayField(features_tensor[hard_negative_index]))
                coords.append(ArrayField(coordinates_tensor[hard_negative_index]))
                masks.append(
                    ArrayField(
                        features_tensor[hard_negative_index].new_ones(
                            (features_tensor[hard_negative_index].shape[0],), dtype=torch.bool
                        ),
                        padding_value=False,
                        dtype=torch.bool,
                    )
                )

            fields: Dict[str, Field] = {
                "caption": TextField(self._tokenizer.tokenize(caption), None),
                "box_features": ListField(features),
                "box_coordinates": ListField(coords),
                "box_mask": ListField(masks),
                "label": LabelField(label, skip_indexing=True),
            }

            return Instance(fields)

    # # todo: implement
    # @overrides
    # def text_to_instance(
    #     self,  # type: ignore
    #     caption: str,
    #     filename: str,
    #     image: Union[str, Tuple[Tensor, Tensor]],
    #     other_images: List[Optional[Union[str, Tuple[Tensor, Tensor]]]],
    #     hard_negatives_cache=Optional[Dict[Tensor, List[Tuple[Tensor, Tensor]]]],
    #     label: int = 0,
    #     *,
    #     use_cache: bool = True,
    # ) -> Optional[Instance]:
    #     caption_field = TextField(self._tokenizer.tokenize(caption), None)

    #     features, coords = self.get_image_features(image)

    #     if self.is_evaluation:
    #         box_features = []
    #         box_coordinates = []
    #         box_masks = []

    #         for curr_image in other_images:
    #             curr_image_features, curr_image_coords = self.get_image_features(curr_image)

    #             box_features.append(ArrayField(curr_image_features))
    #             box_coordinates.append(ArrayField(curr_image_coords))
    #             box_masks.append(
    #                 ArrayField(
    #                     curr_image_features.new_ones(
    #                         (curr_image_features.shape[0],), dtype=torch.bool
    #                     ),
    #                     padding_value=False,
    #                     dtype=torch.bool,
    #                 )
    #             )

    #     else:
    #         hard_negatives = self.get_hard_negatives(
    #             caption, filename, features, other_images, hard_negatives_cache
    #         )

    #         box_features = [
    #             ArrayField(features),
    #             ArrayField(hard_negatives[0][0]),
    #             ArrayField(hard_negatives[1][0]),
    #             ArrayField(hard_negatives[2][0]),
    #         ]

    #         box_coordinates = [
    #             ArrayField(coords),
    #             ArrayField(hard_negatives[0][1]),
    #             ArrayField(hard_negatives[1][1]),
    #             ArrayField(hard_negatives[2][1]),
    #         ]

    #         box_masks = [
    #             ArrayField(
    #                 features.new_ones((features.shape[0],), dtype=torch.bool),
    #                 padding_value=False,
    #                 dtype=torch.bool,
    #             ),
    #             ArrayField(
    #                 hard_negatives[0][0].new_ones(
    #                     (hard_negatives[0][0].shape[0],), dtype=torch.bool
    #                 ),
    #                 padding_value=False,
    #                 dtype=torch.bool,
    #             ),
    #             ArrayField(
    #                 hard_negatives[1][0].new_ones(
    #                     (hard_negatives[1][0].shape[0],), dtype=torch.bool
    #                 ),
    #                 padding_value=False,
    #                 dtype=torch.bool,
    #             ),
    #             ArrayField(
    #                 hard_negatives[2][0].new_ones(
    #                     (hard_negatives[2][0].shape[0],), dtype=torch.bool
    #                 ),
    #                 padding_value=False,
    #                 dtype=torch.bool,
    #             ),
    #         ]

    #     fields: Dict[str, Field] = {
    #         "caption": caption_field,
    #         "box_features": ListField(box_features),
    #         "box_coordinates": ListField(box_coordinates),
    #         "box_mask": ListField(box_masks),
    #         "label": LabelField(label, skip_indexing=True),
    #     }

    #     return Instance(fields)

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

    def get_hard_negatives2(
        self,
        caption: str,
        filename: str,
        image_features: Tensor,
        other_images: List[Optional[Union[str, Tuple[Tensor, Tensor]]]],
        hard_negatives_cache=Dict[Tensor, List[Tuple[Tensor, Tensor]]],
    ) -> List[Tuple[Tensor, Tensor]]:
        image_embedding = torch.mean(image_features, dim=0)
        if filename in hard_negatives_cache:
            return hard_negatives_cache[filename]
        if self.is_test:
            caption_encoding = torch.randn((10))
        # else:
        #     batch = self.tokenizer.encode_plus(caption, return_tensors="pt")
        #     # Shape: (1, 1024)? # TODO: should I squeeze this?
        #     caption_encoding = self.model(**batch).pooler_output.squeeze(0).to(device=self.cuda_device)
        # image_caption_embedding = caption_encoding * image_embedding

        heap = []
        heapq.heapify(heap)
        seen_set = set()
        for image in other_images:
            curr_image_features, curr_image_coords = self.get_image_features(image)
            averaged_features = torch.mean(curr_image_features, dim=0)
            if not torch.equal(averaged_features, image_embedding):
                # Find 3 nearest neighbors
                neg_dist = (
                    -1
                    * torch.dist(
                        # image_caption_embedding, averaged_features * caption_encoding
                        image_embedding,
                        averaged_features,
                    ).item()
                )
                if neg_dist not in seen_set:
                    heapq.heappush(heap, (neg_dist, curr_image_features, curr_image_coords))
                    # TODO: figure out if this heap is working right
                    if len(heap) > 3:
                        heapq.heappop(heap)
                    seen_set.add(neg_dist)

        hard_negative_features = []
        for _, curr_image_features, curr_image_coords in heap:
            hard_negative_features.append((curr_image_features, curr_image_coords))

        hard_negatives_cache[filename] = hard_negative_features

        return hard_negatives_cache[filename]

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

        caption_list = []
        for caption_dict in captions:
            curr_captions = []
            for caption in caption_dict["captions"]:
                # TODO: switch to batch_encode_plus?
                batch = self.tokenizer.encode_plus(caption, return_tensors="pt").to(
                    device=self.cuda_device
                )
                # Shape: (1, 1024)
                caption_embedding = self.model(**batch).pooler_output.squeeze(0)
                curr_captions.append(caption_embedding)
            caption_list.append(torch.stack(curr_captions, dim=0))
        # Shape: (num_captions, 5, 1024)
        return torch.stack(caption_list, dim=0)

    # todo: fix
    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["caption"].token_indexers = self._token_indexers  # type: ignore
