from os import PathLike
from typing import (
    Dict,
    Union,
    Optional,
    Tuple,
    Iterable,
)
import json
import os

from overrides import overrides
import torch
from torch import Tensor
import faiss

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
def get_caption_data(file_path):
    with open(file_path, "r") as f:
        captions = f.read().split("\n")

    image_id = filename.split(".")[0]
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
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
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
        files_in_split = {}
        with open(file_path, "r") as f:
            for line in f:
                files_in_split.add(line)

        logger.info("Reading file at %s", file_path)
        captions = []
        for filename in os.listdir(self.data_dir):  # file_path):
            if filename.split(".")[0] in files_in_split:
                full_file_path = open(os.path.join(self.data_dir, filename))
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
            logger.info("images size: %s", len(self.images))
            try:
                processed_images = self._process_image_paths(
                    self.images[filename] for filename in filenames
                )
            except KeyError as e:
                print(self.images)
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

        for caption_dict, processed_image in zip(caption_dicts, processed_images):
            for caption in caption_dict["captions"]:
                instance = self.text_to_instance(caption, processed_image)
                if instance is not None:
                    yield instance

    # todo: implement
    @overrides
    def text_to_instance(
        self,  # type: ignore
        caption: str,
        image: Optional[Union[str, Tuple[Tensor, Tensor]]],
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

        if image is not None:
            if isinstance(image, str):
                features, coords = next(self._process_image_paths([image], use_cache=use_cache))
            else:
                features, coords = image

            fields["box_features"] = ArrayField(features)
            fields["box_coordinates"] = ArrayField(coords)
            fields["box_mask"] = ArrayField(
                features.new_ones((features.shape[0],), dtype=torch.bool),
                padding_value=False,
                dtype=torch.bool,
            )

        # alternative_features = []
        # alternative_coordinates = []
        # alternative_masks = []
        # if hard_negatives is not None:
        #     for image in hard_negatives:
        #         if isinstance(image, str):
        #             features, coords = next(self._process_image_paths([image], use_cache=use_cache))
        #         else:
        #             features, coords = image

        #         alternative_features.append(TensorField(features))
        #         alternative_coordinates.append(TensorField(coords))
        #         alternative_masks.append(TensorField(
        #             features.new_ones((features.shape[0],), dtype=torch.bool),
        #             padding_value=False,
        #             dtype=torch.bool,
        #         ))
        #     fields["alternative_features"] = ListField(alternative_features)
        #     fields["alternative_coordinates"] = ListField(alternative_coordinates)
        #     fields["alternative_masks"] = ListField(alternative_masks)

        # if label is not None:
        #     if label < 0 or label >= len(sequences):
        #         raise ValueError("Image %d does not exist", label)
        #     from allennlp.data.fields import IndexField

        #     fields["correct_image"] = IndexField(label, sequences)

        # TODO: Should images actually be the labels? Ask Dirk
        # fields["caption"] = LabelField(caption, label_namespace="captions")

        return Instance(fields)

    # todo: fix
    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["caption"].token_indexers = self._token_indexers  # type: ignore
