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
from allennlp.data.fields import ArrayField, LabelField, ListField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector
from allennlp_models.vision.dataset_readers.vision_reader import VisionReader

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

        if not is_test:
            # TODO: do I need the model itself or just the tokenizer?
            self.model = transformers.AutoModel.from_pretrained("bert-large-uncased")
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
        # idea:
        # 1. Read in captions
        # Have a directory with all of the data -> each file corresponds to one image
        # Filter out unrelated captions?
        # 2. Process images
        # 3. Create instances
        # Instance structure:
        # Image id
        # caption tokens (only one caption)
        # correct image
        # 3 hard negatives

        # # TODO: I don't think we need this, because there are test and train/val files
        # # Maybe we need to know how many of the train_and_val are training, and how many are validation
        # # if the splits are using slicing syntax, honor it
        # slice_match = re.match(r"(.*)\[([0123456789:]*)]", file_path)
        # if slice_match is None:
        #     question_slice = slice(None, None, None)
        # else:
        #     split_name = slice_match[1]
        #     slice_args = [int(a) if len(a) > 0 else None for a in slice_match[2].split(":")]
        #     question_slice = slice(*slice_args)

        file_path = cached_path(file_path, extract_archive=True)
        files_in_split = set()
        with open(file_path, "r") as f:
            for line in f:
                files_in_split.add(line.rstrip("\n"))

        # logger.info("Reading file at %s", file_path)
        captions = []
        for filename in os.listdir(self.data_dir):  # file_path):
            if filename.split(".")[0] in files_in_split:
                full_file_path = os.path.join(self.data_dir, filename)
                captions.append(get_caption_data(full_file_path))

        caption_dicts = list(self.shard_iterable(captions))

        # todo: probably move this to a utils file?
        processed_images: Iterable[Optional[Tuple[Tensor, Tensor]]]
        if self.produce_featurized_images:
            # It would be much easier to just process one image at a time, but it's faster to process
            # them in batches. So this code gathers up instances until it has enough to fill up a batch
            # that needs processing, and then processes them all.
            filenames = [f"{caption_dict['image_id']}.jpg" for caption_dict in caption_dicts]
            # for filename in filenames:
            # logger.info("Reading file at %s", filename)
            # logger.info("Reading file at %s", self.images[filename])
            # logger.info("images size: %s", len(self.images))
            try:
                processed_images = list(
                    self._process_image_paths(self.images[filename] for filename in filenames)
                )
            except KeyError as e:
                # print(self.images)
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

        hard_negatives_cache = {}
        image_subset = processed_images[:100]
        for caption_dict, filename, processed_image in zip(caption_dicts, filenames, processed_images):
        # for i in range(len(caption_dicts)):
            # caption_dict = caption_dicts[i]
            # filename = filenames
            for caption in caption_dict["captions"]:
                instance = self.text_to_instance(
                    caption, filename, processed_image, image_subset, hard_negatives_cache
                )
                print(len(hard_negatives_cache))
                if instance is not None:
                    yield instance

    # todo: implement
    @overrides
    def text_to_instance(
        self,  # type: ignore
        caption: str,
        filename: str,
        image: Optional[Union[str, Tuple[Tensor, Tensor]]],
        other_images: List[Optional[Union[str, Tuple[Tensor, Tensor]]]],
        hard_negatives_cache=Dict[Tensor, List[Tuple[Tensor, Tensor]]],
        # images: Optional[List[Union[str, Tuple[Tensor, Tensor]]]],
        # label: Optional[int] = None,
        *,
        use_cache: bool = True,
    ) -> Optional[Instance]:
        caption_field = TextField(self._tokenizer.tokenize(caption), None)

        fields: Dict[str, Field] = {
            "caption": caption_field,
        }

        if image is None:
            return None

        features, coords = self.get_image_features(image)

        # print("features device:")
        # print(features.device)
        hard_negatives = self.get_hard_negatives(
            caption, filename, features, other_images, hard_negatives_cache
        )

        # fields["box_features"] = ArrayField(features)
        # fields["box_coordinates"] = ArrayField(coords)
        # fields["box_mask"] = ArrayField(
        #     features.new_ones((features.shape[0],), dtype=torch.bool),
        #     padding_value=False,
        #     dtype=torch.bool,
        # )

        fields["box_features"] = ListField(
            [
                ArrayField(features),
                ArrayField(hard_negatives[0][0]),
                ArrayField(hard_negatives[1][0]),
                ArrayField(hard_negatives[2][0]),
            ]
        )
        fields["box_coordinates"] = ListField(
            [
                ArrayField(coords),
                ArrayField(hard_negatives[0][1]),
                ArrayField(hard_negatives[1][1]),
                ArrayField(hard_negatives[2][1]),
            ]
        )
        fields["box_mask"] = ListField(
            [
                ArrayField(
                    features.new_ones((features.shape[0],), dtype=torch.bool),
                    padding_value=False,
                    dtype=torch.bool,
                ),
                ArrayField(
                    hard_negatives[0][0].new_ones(
                        (hard_negatives[0][0].shape[0],), dtype=torch.bool
                    ),
                    padding_value=False,
                    dtype=torch.bool,
                ),
                ArrayField(
                    hard_negatives[1][0].new_ones(
                        (hard_negatives[1][0].shape[0],), dtype=torch.bool
                    ),
                    padding_value=False,
                    dtype=torch.bool,
                ),
                ArrayField(
                    hard_negatives[2][0].new_ones(
                        (hard_negatives[2][0].shape[0],), dtype=torch.bool
                    ),
                    padding_value=False,
                    dtype=torch.bool,
                ),
            ]
        )

        fields["label"] = LabelField(0, skip_indexing=True)

        return Instance(fields)

    def get_hard_negatives(
        self,
        caption: str,
        filename: str,
        image_features: Tensor,
        other_images: List[Optional[Union[str, Tuple[Tensor, Tensor]]]],
        hard_negatives_cache=Dict[Tensor, List[Tuple[Tensor, Tensor]]],
    ) -> List[Tuple[Tensor, Tensor]]:
        # print("features device2:")
        # print(image_features.device)
        # print("image embedding device:")
        image_embedding = torch.mean(image_features, dim=0)
        # print(image_embedding.device)
        if filename in hard_negatives_cache:
            return hard_negatives_cache[filename]
        if self.is_test:
            caption_encoding = torch.randn((10))
        else:
            batch = self.tokenizer.encode_plus(caption, return_tensors="pt")
            # Shape: (1, 1024)? # TODO: should I squeeze this?
            caption_encoding = self.model(**batch).pooler_output.squeeze(0).to(device=self.cuda_device)
        image_caption_embedding = caption_encoding * image_embedding

        heap = []
        heapq.heapify(heap)
        # TODO: see if we don't have to sample? the cache might not be working for hard negatives
        # print(len(other_images))
        # sampled_images = choices(other_images, k=min(len(other_images), 500))
        # print(len(sampled_images))
        for image in other_images: # sample(other_images, min(len(other_images), 100)):
            # Calculate the 3 closest hard negatives:
            # 1. Calculate mean of all boxes
            # 2. Find the ~100 nearest neighbors of the input image
            curr_image_features, curr_image_coords = self.get_image_features(image)
            averaged_features = torch.mean(curr_image_features, dim=0)
            if not torch.equal(averaged_features, image_embedding):
                # Find 3 nearest neighbors
                neg_dist = (
                    -1
                    * torch.dist(
                        image_caption_embedding, averaged_features * caption_encoding
                        # image_embedding, averaged_features
                    ).item()
                )
                heapq.heappush(heap, (neg_dist, curr_image_features, curr_image_coords))
                if len(heap) > 3:
                    heapq.heappop(heap)

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
        return features.to(device=self.cuda_device), coords.to(device=self.cuda_device)

    # todo: fix
    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["caption"].token_indexers = self._token_indexers  # type: ignore
